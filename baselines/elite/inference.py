import os
import sys
import yaml
import hashlib
import argparse
from types import MethodType
from functools import partial
from typing import Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import Attention
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

sys.path.append('./../../')
from nb_utils.configs import classes_data
from persongen.utils.seed import fix_seed

from baselines.elite.datasets import CustomDatasetWithBG

from baselines.elite.train_local import Mapper, MapperLocal
from baselines.elite.train_local import inj_forward_text, inj_forward_crossattention, validation


def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


def pww_load_tools(
    device: str = "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    mapper_model_path: Optional[str] = None,
    mapper_local_model_path: Optional[str] = None,
    diffusion_model_path: Optional[str] = None,
    model_token: Optional[str] = None,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    CLIPVisionModel,
    Mapper,
    MapperLocal,
    LMSDiscreteScheduler,
]:

    # 'CompVis/stable-diffusion-v1-4'
    # local_path_only = diffusion_model_path is not None
    local_path_only = False
    vae = AutoencoderKL.from_pretrained(
        diffusion_model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    from transformers.modeling_utils import logger
    logger.disabled = True
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    logger.disabled = False

    # Load models and create wrapper for stable diffusion
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.forward = MethodType(inj_forward_text, _module)

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    mapper = Mapper(input_dim=1024, output_dim=768)

    mapper_local = None
    if mapper_local_model_path is not None:
        mapper_local = MapperLocal(input_dim=1024, output_dim=768)

    for _name, _module in unet.named_modules():
        if isinstance(_module, Attention) and _module.is_cross_attention:
            _module.forward = MethodType(partial(inj_forward_crossattention, use_local=mapper_local is not None), _module)

            shape = _module.to_k.weight.shape
            to_k_global = torch.nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = torch.nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            if mapper_local is not None:
                to_v_local = torch.nn.Linear(shape[1], shape[0], bias=False)
                mapper_local.add_module(f'{_name.replace(".", "_")}_to_v', to_v_local)

                to_k_local = torch.nn.Linear(shape[1], shape[0], bias=False)
                mapper_local.add_module(f'{_name.replace(".", "_")}_to_k', to_k_local)

    mapper.load_state_dict(torch.load(mapper_model_path, map_location='cpu', weights_only=True))
    mapper.half()

    if mapper_local is not None:
        mapper_local.load_state_dict(torch.load(mapper_local_model_path, map_location='cpu', weights_only=True))
        mapper_local.half()

    for _name, _module in unet.named_modules():
        if isinstance(_module, Attention) and _module.is_cross_attention:
            _module.add_module('to_k_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))

            if mapper_local is not None:
                _module.add_module('to_v_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_v'))
                _module.add_module('to_k_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_k'))

    vae.to(device), unet.to(device), text_encoder.to(device), image_encoder.to(device), mapper.to(device), mapper_local.to(device) if mapper_local is not None else None

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()
    mapper.eval()
    if mapper_local is not None:
        mapper_local.eval()
    return vae, unet, text_encoder, tokenizer, image_encoder, mapper, mapper_local if mapper_local is not None else None, scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--global_mapper_path",
        type=str,
        required=True,
        help="Path to pretrained global mapping network.",
    )

    parser.add_argument(
        "--local_mapper_path",
        type=str,
        required=False,
        default=None,
        help="Path to pretrained local mapping network.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs',
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="S",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--template",
        type=str,
        default="a photo of a {}",
        help="Text template for customized genetation.",
    )

    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=True, help="A folder containing the testing data."
    )
    parser.add_argument("--class_name", type=str, default=None)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--token_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--selected_data",
        type=int,
        default=-1,
        help="Data index. -1 for all.",
    )

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--guidance_scale_ref",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--change_step",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--llambda",
        type=float,
        default=0.8,
        help="Lambda for fuse the global and local feature.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for testing.",
    )

    parser.add_argument(
        "--create_exp_dir",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.global_mapper_path):
        print('Downloading Global Mapper weights...')
        os.makedirs(os.path.dirname(args.global_mapper_path), exist_ok=True)
        torch.hub.download_url_to_file(
            'https://nxt.2a2i.org/index.php/s/os6MLWLxbDCFGwZ/download', args.global_mapper_path
        )

    if args.local_mapper_path is not None and not os.path.exists(args.local_mapper_path):
        print('Downloading Local Mapper weights...')
        os.makedirs(os.path.dirname(args.local_mapper_path), exist_ok=True)
        torch.hub.download_url_to_file(
            'https://nxt.2a2i.org/index.php/s/bryMpwHWPmpnntc/download', args.local_mapper_path
        )

    vae, unet, text_encoder, tokenizer, image_encoder, mapper, mapper_local, scheduler = pww_load_tools(
            "cuda:0",
            LMSDiscreteScheduler,
            diffusion_model_path=args.pretrained_model_name_or_path,
            mapper_model_path=args.global_mapper_path,
            mapper_local_model_path=args.local_mapper_path,
        )

    args.test_data_dir = os.path.abspath(args.test_data_dir)
    concept_name = os.path.basename(args.test_data_dir)
    assert (concept_name in classes_data) ^ (args.class_name is not None), 'Either class_name or test_data_dir should define concept name'
    args.class_name = args.class_name or classes_data[concept_name][-1]

    if args.create_exp_dir:
        exp_name = f'00000-{hashlib.sha256(concept_name.encode("utf-8")).hexdigest()[-4:]}-{concept_name}'
        args.output_dir = os.path.join(args.output_dir, exp_name)

        config = {
            'resolution': 512,
            'placeholder_token': args.placeholder_token,
            'output_dir': args.output_dir,
            'exp_name': exp_name,
            'class_name': args.class_name,
            'test_data_dir': args.test_data_dir.replace('baselines/elite/dreambooth', 'dreambooth/dataset'),
        }

        os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
        with open(os.path.join(args.output_dir, 'logs', "hparams.yml"), "w") as outfile:
            yaml.dump(config, outfile)

        args.output_dir = os.path.join(
            args.output_dir, 'checkpoint-0', 'samples',
            f'ns{args.num_inference_steps}_gs{args.guidance_scale}_sg{args.guidance_scale_ref}_chs{args.change_step}_llambda{args.llambda}' +
            ('_LM' if mapper_local is not None else '_GM'), 'version_0'
        )

    os.makedirs(args.output_dir, exist_ok=True)

    for template in args.template.split('#'):
        train_dataset = CustomDatasetWithBG(
            data_root=args.test_data_dir,
            tokenizer=tokenizer,
            size=512,
            placeholder_token=args.placeholder_token,
            template=template,
            class_name=args.class_name
        )
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        for step, batch in enumerate(train_dataloader):
            batch: dict
            if -1 < args.selected_data != step:
                continue
            batch["pixel_values"] = batch["pixel_values"].to("cuda:0")
            batch["pixel_values_clip"] = batch["pixel_values_clip"].to("cuda:0").half()
            batch["pixel_values_obj"] = batch["pixel_values_obj"].to("cuda:0").half()
            batch["pixel_values_seg"] = batch["pixel_values_seg"].to("cuda:0").half()
            batch["input_ids"] = batch["input_ids"].to("cuda:0")
            batch["index"] = batch["index"].to("cuda:0").long()
            if 'input_ids_superclass' in batch:
                batch["input_ids_superclass"] = batch["input_ids_superclass"].to("cuda:0")
            print(step, batch['text'], batch["text_superclass"])

            os.makedirs(os.path.join(args.output_dir, batch['text'][0]), exist_ok=True)

            generator = fix_seed(batch['text'][0], args.seed, unet.device)
            shape = (args.num_images_per_prompt, unet.in_channels, 64, 64)
            latents = randn_tensor(shape, generator=generator, dtype=unet.dtype, device=unet.device)

            for idx in range(args.num_images_per_prompt):
                if os.path.exists(os.path.join(args.output_dir, batch['text'][0], f'{idx}.png')):
                    continue

                [syn_image] = validation(
                    batch, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae,
                    batch["pixel_values_clip"].device, latents=latents[idx: idx + 1],
                    guidance_scale=args.guidance_scale, guidance_scale_ref=args.guidance_scale_ref, change_step=args.change_step,
                    token_index=args.token_index, num_steps=args.num_inference_steps, seed=None, llambda=args.llambda
                )
                syn_image.save(os.path.join(args.output_dir, batch['text'][0], f'{idx}.png'))


if __name__ == "__main__":
    main(parse_args())
