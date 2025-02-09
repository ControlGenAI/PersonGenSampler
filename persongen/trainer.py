import os
from typing import Callable, Any, Optional

import yaml
import random
import secrets
import logging
import itertools
from collections import defaultdict

import tqdm.autonotebook as tqdm
import wandb

import numpy as np

import torch
from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, Transformer2DModel, PixArtAlphaPipeline
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
import transformers.utils.logging
from transformers import T5EncoderModel, T5Tokenizer

from nb_utils.eval_sets import small_sets
from nb_utils.configs import live_object_data
from nb_utils.images_viewer import MultifolderViewer
from nb_utils.clip_eval import ExpEvaluator, aggregate_similarities

from .data.dataset import ImageDataset, DreamBoothDataset, collate_fn
from .utils.registry import ClassRegistry
from .model.svd_diff import setup_module_for_svd_diff
from .model.utils import count_trainable_params, params_grad_norm


logger = get_logger(__name__)
trainers = ClassRegistry()

torch.backends.cuda.enable_flash_sdp(True)


@trainers.add_to_registry('base')
class BaseTrainer:
    BASE_PROMPT = "a photo of {0}"

    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(dict)

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.train_data_dir))
        )
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Get the last experiment idx
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            # noinspection PyBroadException
            try:
                curr_exp_idx = max(exp_idx, int(folder.split('-')[0].lstrip('0')))
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        self.config.exp_name = self.setup_exp_name(exp_idx)

        self.config.output_dir = os.path.abspath(os.path.join(self.config.output_dir, self.config.exp_name))

        if os.path.exists(self.config.output_dir):
            raise ValueError(f'Experiment directory {self.config.output_dir} already exists. Race condition!')
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.config.logging_dir = os.path.join(self.config.output_dir, 'logs')
        os.makedirs(self.config.logging_dir, exist_ok=True)

        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        if self.config.api_key is not None:
            wandb.login(key=self.config.api_key)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir)
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            log_with='wandb',
            project_config=accelerator_project_config,
        )
        self.accelerator.init_trackers(
            project_name=self.config.project_name,
            config=self.config,
            init_kwargs={"wandb": {
                "name": self.config.exp_name,
                'settings': wandb.Settings(code_dir=os.path.dirname(self.config.argv[1]))
            }}
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler", revision=self.config.revision
        )
        self.transformer = Transformer2DModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder='transformer', revision=self.config.revision
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder", revision=self.config.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="vae", revision=self.config.revision
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer", revision=self.config.revision
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.params_to_optimize = []

        if self.config.finetune_text_encoder:
            self.text_encoder.train()
            if self.config.qkv_only:
                te_params_to_optimize = []
                for (name, param) in self.text_encoder.named_parameters():
                    if 'SelfAttention.q' in name or 'SelfAttention.k' in name or 'SelfAttention.v' in name:
                        param.requires_grad = True
                        te_params_to_optimize.append(param)
            else:
                self.text_encoder.requires_grad_(True)
                te_params_to_optimize = list(self.text_encoder.parameters())
            self.params_to_optimize.append({'name': 'text_encoder', 'params': te_params_to_optimize})

        if self.config.finetune_transformer:
            self.transformer.train()
            if self.config.qkv_only:
                transformer_params_to_optimize = []
                for (name, param) in self.transformer.named_parameters():
                    if 'to_q' in name or 'to_k' in name or 'to_v' in name:
                        param.requires_grad = True
                        transformer_params_to_optimize.append(param)
            else:
                self.transformer.requires_grad_(True)
                transformer_params_to_optimize = list(self.transformer.parameters())
            self.params_to_optimize.append({'name': 'transformer', 'params': transformer_params_to_optimize})

        transformer_numel = count_trainable_params(self.transformer, verbose=True)
        te_numel = count_trainable_params(self.text_encoder, verbose=True)

        print(
            '# params Transformer: {0}, # params TE: {1}, # params: {2}'.format(
                transformer_numel, te_numel, transformer_numel + te_numel
            )
        )

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt=BaseTrainer.BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}'),
                class_data_root=self.config.class_data_dir if self.config.with_prior_preservation else None,
                class_prompt=BaseTrainer.BASE_PROMPT.format(self.config.class_name),
                tokenizer=self.tokenizer,
            )
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
                examples, self.config.with_prior_preservation
            )
        else:
            self.train_dataset = ImageDataset(
                train_data_dir=self.config.train_data_dir
            )
            collator = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator
        )

    # noinspection PyTypeChecker
    def move_to_device(self):
        self.optimizer, self.train_dataloader = self.accelerator.prepare(self.optimizer, self.train_dataloader)
        if self.config.finetune_transformer:
            self.transformer = self.accelerator.prepare(self.transformer)
        if self.config.finetune_text_encoder:
            self.text_encoder = self.accelerator.prepare(self.text_encoder)

        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        # All trained parameters should be explicitly moved to float32 even for mixed precision training
        if self.config.finetune_transformer:
            for param in self.transformer.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)
        if self.config.finetune_text_encoder:
            for param in self.text_encoder.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()
        self.setup_evaluator()

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(batch['pixel_values'].to(self.weight_dtype)).latent_dist.sample()
        else:
            latents = self.vae.encode(batch['image'].to(self.weight_dtype) * 2.0 - 1.0).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation:
            inputs = self.tokenizer(
                BaseTrainer.BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}'),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device=latents.device)
            input_attn_masks = inputs["attention_mask"].to(device=latents.device)
            encoder_hidden_states = self.text_encoder(input_ids, attention_mask=input_attn_masks)[0]
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(latents.shape[0], 0)
        else:
            input_attn_masks = batch['input_attn_masks']
            encoder_hidden_states = self.text_encoder(batch["input_ids"], attention_mask=input_attn_masks)[0]

        # Note: encoder_hidden_states contains embeddings for the following combinations of tokens:
        # if with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'CLASS_NAME</w>',                          '<|endoftext|>', ...PADDING]
        # if not with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        outputs = self.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=input_attn_masks,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs
        ).sample.chunk(2, 1)[0]

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(prior_outputs.float(), prior_target.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss

    def setup_pipeline(self):
        self.pipeline = PixArtAlphaPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            tokenizer=self.tokenizer,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder, keep_fp32_wrapper=False),
            transformer=self.accelerator.unwrap_model(self.transformer, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            requires_safety_checker=False,
            torch_dtype=torch.float16
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    # noinspection PyPep8Naming
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = small_sets[live_object_data[self.config.class_name]]

        samples_path = os.path.join(
            self.config.output_dir, f'checkpoint-{epoch}', 'samples',
            'ns0_gs0_validation', 'version_0'
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            caption = prompt.format(f'{self.config.placeholder_token} {self.config.class_name}')
            kwargs = {
                "num_inference_steps": 20,
                "guidance_scale": 4.5,
                "prompt": caption,
                "num_images_per_prompt": self.config.num_val_imgs
            }
            images = self.pipeline(generator=generator, **kwargs).images

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f'{idx}.png'))

        validation_results = self._evaluate(samples_path)

        IS_key = f'small/{live_object_data[self.config.class_name]}_with_class_image_similarity'
        TS_key = f'small/{live_object_data[self.config.class_name]}_with_class_text_similarity'

        IS, TS = validation_results[IS_key], validation_results[TS_key]

        self.metrics[epoch] |= {'IS': IS, 'TS': TS}
        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    "val_is": IS,
                    "val_ts": TS,
                    "val_metrics": 2 / (1 / IS + 1 / TS),
                    'validation': [
                        wandb.Image(image, caption=caption) for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)

        if self.config.finetune_text_encoder:
            torch.save(self.text_encoder.state_dict(), os.path.join(save_path, 'text_encoder.bin'))
        if self.config.finetune_transformer:
            torch.save(self.transformer.state_dict(), os.path.join(save_path, 'transformer.bin'))

    def train(self):
        if self.accelerator.is_main_process:
            self.validation(0)
            self.save_model(0)

        for epoch in tqdm.tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            with self.accelerator.autocast():
                loss = self.train_step(batch)
            self.accelerator.backward(loss)

            for tracker in self.accelerator.trackers:
                tracker.log({
                    "loss": loss,
                    "transformer_grad_norm": params_grad_norm(self.transformer.parameters()),
                    "text_encoder_grad_norm": params_grad_norm(self.text_encoder.parameters())
                } | {
                    f'group_{group["name"]}_lr': group['lr']
                    for group in self.optimizer.param_groups
                } | {
                    f'group_{group["name"]}_numel': sum(_.numel() for _ in group['params'])
                    for group in self.optimizer.param_groups
                })

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch != 0:
                    self.validation(epoch)
                    self.save_model(epoch)

        if self.accelerator.is_main_process:
            self.validation(self.config.num_train_epochs)
            self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()

    def setup_evaluator(self):
        self.evaluator = ExpEvaluator(self.accelerator.device)

    def _evaluate(self, path):
        viewer = MultifolderViewer(path, lazy_load=False)

        results = self.evaluator(viewer, vars(self.config))
        results |= {'config': vars(self.config)}
        results |= aggregate_similarities(results)

        return results


@trainers.add_to_registry('svd')
class SVDDiffTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        assert self.config.finetune_transformer

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.train_data_dir))
        )
        exp_name += '_SVDDiff'
        return exp_name

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.params_to_optimize = []

        if self.config.finetune_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            setup_module_for_svd_diff(self.text_encoder, scale=1.0, deltas_path=None, qkv_only=self.config.qkv_only)
            self.text_encoder.train()

        if self.config.finetune_transformer:
            self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
            setup_module_for_svd_diff(self.transformer, scale=1.0, deltas_path=None, qkv_only=self.config.qkv_only)
            self.transformer.train()

        delta_params_to_optimize, delta_params_to_optimize_1d = [], []
        for name, param in itertools.chain(self.transformer.named_parameters(), self.text_encoder.named_parameters()):
            if not param.requires_grad:
                continue
            if self.config.learning_rate_1d is not None and 'norm' in name:
                delta_params_to_optimize_1d.append(param)
            else:
                delta_params_to_optimize.append(param)

        self.params_to_optimize.append(
            {'name': 'transformer+text_encoder', "params": delta_params_to_optimize}
        )
        self.params_to_optimize.append(
            {'name': 'transformer+text_encoder 1d', "params": delta_params_to_optimize_1d, "lr": self.config.learning_rate_1d}
        )

        transformer_numel = count_trainable_params(self.transformer, verbose=True)
        te_numel = count_trainable_params(self.text_encoder, verbose=True)

        print(
            '# params Transformer: {0}, # params TE: {1}, # params: {2}'.format(
                transformer_numel, te_numel, transformer_numel + te_numel
            )
        )

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)

        if self.config.finetune_text_encoder:
            torch.save(
                {name: param for name, param in self.text_encoder.state_dict().items() if '.delta' in name},
                os.path.join(save_path, 'text_encoder.bin')
            )

        if self.config.finetune_transformer:
            torch.save(
                {name: param for name, param in self.transformer.state_dict().items() if '.delta' in name},
                os.path.join(save_path, 'transformer.bin')
            )
