import os
from typing import List

import tqdm.autonotebook as tqdm

import torch
from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel
)
from diffusers.utils.torch_utils import randn_tensor

from transformers import CLIPTextModel, CLIPTokenizer

from .model.pipeline import (
    StableDiffusionPipelineMultiStage,
    StableDiffusionPipelinePhotoswap,
    StableDiffusionPipelineCrossAttnMasked,
)
from .model.svd_diff import setup_module_for_svd_diff
from .utils.registry import ClassRegistry
from .utils.seed import fix_seed

from baselines.custom_diffusion.train_custom_diffusion import setup_unet_for_custom_diffusion

inferencers = ClassRegistry()


@inferencers.add_to_registry('base')
class BaseInferencer:
    def __init__(
        self, config, args, context_prompts, base_prompts,
        dtype=torch.float32, device=torch.device('cuda', 0)
    ):
        self.config = config
        self.args = args
        self.checkpoint_idx = args.checkpoint_idx
        self.num_images_per_context_prompt = args.num_images_per_medium_prompt
        self.num_images_per_base_prompt = args.num_images_per_base_prompt
        self.batch_size_context = args.batch_size_medium
        self.batch_size_base = args.batch_size_base

        config['output_dir'] = args.output_dir or config['output_dir']

        if self.checkpoint_idx is None:
            self.checkpoint_path = config['output_dir']
        else:
            self.checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{self.checkpoint_idx}')

        self.context_prompts = context_prompts
        self.base_prompts = base_prompts

        self.replace_inference_output = self.args.replace_inference_output
        self.version = self.args.version

        self.device = device
        self.dtype = dtype

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'num_inference_steps': self.args.num_inference_steps,
        }

    def setup_base_model(self):
        # Here we create base models
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], torch_dtype=self.dtype,
            subfolder="scheduler", revision=self.config['revision']
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], torch_dtype=self.dtype,
            subfolder="unet", revision=self.config['revision']
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'], torch_dtype=self.dtype,
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'], torch_dtype=self.dtype,
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], torch_dtype=self.dtype,
            subfolder="text_encoder", revision=self.config['revision']
        )

        self._base_init_pipe_kwargs = {
            'pretrained_model_name_or_path': self.config['pretrained_model_name_or_path'],
            'revision':                      self.config.get('revision'),
            'vae':                           self.vae,
            'text_encoder':                  self.text_encoder,
            'tokenizer':                     self.tokenizer,
            'unet':                          self.unet,
            'scheduler':                     self.scheduler,
            'requires_safety_checker':       False,
            'torch_dtype':                   self.dtype
        }

    def setup_model(self):
        # Here we change some base models if necessary
        pass

    def setup_pipeline(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(**self._base_init_pipe_kwargs)

        self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

    def setup(self):
        self.setup_base_model()
        self.setup_model()
        self.setup_pipeline()
        self.setup_pipe_kwargs()
        self.create_folder_name()
        self.setup_paths()

    def prepare_prompts(self, prompts: List[str]) -> List[dict]:
        prepared_prompts = []
        for prompt in prompts:
            prepared_prompts.append({
                'generate': {
                    'empty_prompt': prompt,
                    'prompt': prompt.format(self.config['class_name'])
                },
                'formatted': prompt.format(self.config['class_name'])
            })

        return prepared_prompts

    def create_folder_name(self):
        self.inference_folder_name = f'ns{self.pipe_kwargs["num_inference_steps"]}_gs{self.pipe_kwargs["guidance_scale"]}'

    def setup_paths(self):
        if self.version is None:
            version = 0
            while True:
                samples_path = os.path.join(
                    self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
                )
                if not os.path.exists(samples_path):
                    break
                version += 1
        else:
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{self.version}'
            )
        self.samples_path = samples_path

    def check_generation(self, path, num_images_per_prompt):
        if self.replace_inference_output:
            return True
        else:
            if os.path.exists(path) and len(os.listdir(path)) == num_images_per_prompt:
                return False
            else:
                return True

    def generate_with_prompt(self, prompt, num_images_per_prompt, batch_size):
        batch_size = max(1, min(batch_size, num_images_per_prompt))
        n_batches = (num_images_per_prompt - 1) // batch_size + 1
        images = []
        generator = fix_seed(prompt.pop('empty_prompt'), self.args.seed, self.device)
        shape = (
            num_images_per_prompt, self.pipe.unet.config.in_channels,
            self.pipe.unet.config.sample_size, self.pipe.unet.config.sample_size
        )
        latents = randn_tensor(shape, generator=generator, dtype=self.pipe.unet.dtype, device=self.pipe.unet.device)
        for i in range(n_batches):
            latents_batch = latents[i * batch_size: (i + 1) * batch_size]
            images_batch = self.pipe(
                **prompt,
                num_images_per_prompt=latents_batch.shape[0],
                generator=generator,
                latents=latents_batch,
                **self.pipe_kwargs
            ).images
            images += images_batch
        return images

    @staticmethod
    def save_images(images, path):
        os.makedirs(path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f'{idx}.png'))

    def generate_with_prompt_list(self, prompts, num_images_per_prompt, batch_size):
        for prompt in tqdm.tqdm(prompts):
            path = os.path.join(self.samples_path, prompt['formatted'])
            if self.check_generation(path, num_images_per_prompt):
                images = self.generate_with_prompt(
                    prompt['generate'], num_images_per_prompt, batch_size
                )
                self.save_images(images, path)

    def generate(self):
        self.generate_with_prompt_list(
            self.prepare_prompts(self.context_prompts), self.num_images_per_context_prompt, self.batch_size_context
        )
        self.generate_with_prompt_list(
            self.prepare_prompts(self.base_prompts), self.num_images_per_base_prompt, self.batch_size_base
        )


@inferencers.add_to_registry('svd_base')
class SVDDiffBaseInferencer(BaseInferencer):
    def setup_model(self):
        if self.config['finetune_text_encoder']:
            self.text_encoder.to(self.device)
            setup_module_for_svd_diff(
                self.text_encoder, scale=self.args.svd_scale, qkv_only=self.config['qkv_only'],
                deltas_path=os.path.join(self.checkpoint_path, 'text_encoder.bin'), fuse=True
            )

        if self.config['finetune_unet']:
            self.unet.to(self.device)
            setup_module_for_svd_diff(
                self.unet, scale=self.args.svd_scale, qkv_only=self.config['qkv_only'],
                deltas_path=os.path.join(self.checkpoint_path, 'unet.bin'), fuse=True
            )

    def create_folder_name(self):
        super().create_folder_name()
        self.inference_folder_name += f'_scale{self.args.svd_scale}'


@inferencers.add_to_registry('superclass')
class SuperclassInferencer(BaseInferencer):
    def create_folder_name(self):
        self.inference_folder_name = f'ns{self.pipe_kwargs["num_inference_steps"]}_gs{self.pipe_kwargs["guidance_scale"]}_superclass'


@inferencers.add_to_registry('multistage')
class MultiStageInferencer(BaseInferencer):
    def setup_model(self):
        is_textual_inversion = self.config['exp_name'].endswith('TI')
        is_custom_diffusion = self.config['exp_name'].endswith('CD')

        unet_path = os.path.join(self.checkpoint_path, 'unet.bin')
        te_path = os.path.join(self.checkpoint_path, 'text_encoder.bin')

        if is_textual_inversion:
            assert os.path.exists(te_path) and not os.path.exists(unet_path), 'TI should load only textual encoder'
        elif is_custom_diffusion:
            assert os.path.exists(te_path) and os.path.exists(unet_path), 'CD should load both textual encoder and UNet'

        if is_textual_inversion or is_custom_diffusion:
            placeholder_tokens = [self.config['placeholder_token']]
            _ = self.tokenizer.add_tokens(placeholder_tokens)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        if is_custom_diffusion:
            setup_unet_for_custom_diffusion(
                self.unet,
                train_kv=True, train_q_out=False if self.config['freeze_model'] == "crossattn_kv" else True,
                enable_xformers_memory_efficient_attention=False
            )

        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, weights_only=True))

        if os.path.exists(te_path):
            _ = self.text_encoder.load_state_dict(torch.load(te_path, weights_only=True), strict=False)
            # Validate changes in buffers layout since transformers>=4.31.0
            assert _.missing_keys == ['text_model.embeddings.position_ids']

    def setup_pipeline(self):
        self.pipe = StableDiffusionPipelineMultiStage.from_pretrained(**self._base_init_pipe_kwargs)

        self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'guidance_scale_ref': self.args.guidance_scale_ref,
            'num_inference_steps': self.args.num_inference_steps,
            'change_step': self.args.change_step,
        }

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_sg{self.args.guidance_scale_ref}_chs{self.args.change_step}"

    def prepare_prompts(self, prompts: List[str]) -> List[dict]:
        holder = self.config['placeholder_token']
        if self.args.with_class_name:
            holder = '{0} {1}'.format(holder, self.config['class_name'])

        prepared_prompts = []
        for prompt in prompts:
            prepared_prompts.append({
                'generate': {
                    'empty_prompt': prompt,
                    'prompt': prompt.format(holder),
                    'ref_prompt': prompt.format(self.config['class_name'])
                },
                'formatted': prompt.format(holder)
            })

        return prepared_prompts


@inferencers.add_to_registry('photoswap')
class PhotoswapInferencer(MultiStageInferencer):
    def setup_pipeline(self):
        self.pipe = StableDiffusionPipelinePhotoswap.from_pretrained(**self._base_init_pipe_kwargs)

        self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'guidance_scale_ref': self.args.guidance_scale_ref,
            'num_inference_steps': self.args.num_inference_steps,
            'self_attn_feat_step': self.args.photoswap_sf_step,
            'cross_attn_map_step': self.args.photoswap_cm_step,
            'self_attn_map_step': self.args.photoswap_sm_step
        }

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_sg{self.args.guidance_scale_ref}_photoswap_sfs{self.args.photoswap_sf_step}_cms{self.args.photoswap_cm_step}_sms{self.args.photoswap_sm_step}"


@inferencers.add_to_registry('crossattn_masked')
class CrossAttnMaskedInferencer(MultiStageInferencer):
    def setup_pipeline(self):
        self.pipe = StableDiffusionPipelineCrossAttnMasked.from_pretrained(**self._base_init_pipe_kwargs)

        self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'num_inference_steps': self.args.num_inference_steps,
            'return_one_image': True,
            'change_step': self.args.change_step,
            'quantile': self.args.quantile,
            'inner_sg': (self.args.inner_gs_1, self.args.inner_gs_2),
            'out_sg': (self.args.out_gs_1, self.args.out_gs_2),
        }

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"
        self.inference_folder_name += f'_chs{self.args.change_step}_quantile{self.args.quantile}'
        self.inference_folder_name += f'_innersg{self.args.inner_gs_1}{self.args.inner_gs_2}'
        self.inference_folder_name += f'_outsg{self.args.out_gs_1}{self.args.out_gs_2}'

        self.inference_folder_name += '_crossattn'


@inferencers.add_to_registry('svd_superclass')
class SVDDiffSuperclassInferencer(SVDDiffBaseInferencer, SuperclassInferencer):
    pass


@inferencers.add_to_registry('svd_multistage')
class SVDDiffMultiStageInferencer(SVDDiffBaseInferencer, MultiStageInferencer):
    pass


@inferencers.add_to_registry('svd_photoswap')
class SVDDiffPhotoswapInferencer(SVDDiffBaseInferencer, PhotoswapInferencer):
    pass


@inferencers.add_to_registry('svd_crossattn_masked')
class SVDDiffCrossAttnMaskedInferencer(SVDDiffBaseInferencer, CrossAttnMaskedInferencer):
    pass
