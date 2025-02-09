import os
import sys
import tqdm
import yaml
import argparse
from typing import Union, LiteralString

import torch
import torch.backends.cuda

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append('./../../')
from nb_utils.configs import live_object_data
from nb_utils.eval_sets import base_set, live_set, object_set

from baselines.custom_diffusion.train_custom_diffusion import setup_unet_for_custom_diffusion

from persongen.utils.seed import fix_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Do not perform actual inference. Only show what prompts will be used for inference"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to hparams.yml"
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        required=False,
        help="Adjust target prompts with class name"
    )
    parser.add_argument(
        "--num_images_per_base_prompt",
        type=int,
        default=30,
        help="Number of generated images for each prompt",
    )
    parser.add_argument(
        "--num_images_per_medium_prompt",
        type=int,
        default=10,
        help="Number of generated images for each prompt",
    )
    parser.add_argument(
        "--batch_size_medium",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size_base",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    return parser.parse_args()


def main(args):
    if 'V100' in torch.cuda.get_device_name(torch.device('cuda')):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    with open(args.config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    exp_path = config['output_dir']

    if live_object_data[config['class_name']] == 'live':
        medium_set = live_set
    else:
        medium_set = object_set

    if args.checkpoint_idx is None:
        checkpoint_path = exp_path
    else:
        checkpoint_path = os.path.join(exp_path, f'checkpoint-{args.checkpoint_idx}')

    unet = UNet2DConditionModel.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="unet", revision=config['revision']
    )
    setup_unet_for_custom_diffusion(
        unet, train_kv=True, train_q_out=False if config['freeze_model'] == "crossattn_kv" else True,
        enable_xformers_memory_efficient_attention=False
    )
    unet.load_state_dict(torch.load(os.path.join(checkpoint_path, 'unet.bin')))
    tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_name_or_path'], subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="text_encoder", revision=config['revision']
    )
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.load_state_dict(torch.load(
        os.path.join(checkpoint_path, 'text_encoder.bin')
    ))
    scheduler = DDIMScheduler.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        config['pretrained_model_name_or_path'],
        tokenizer=tokenizer,
        unet=unet,
        text_encoder=text_encoder,
        scheduler=scheduler,
        torch_dtype=torch.float32).to("cuda")
    pipe.safety_checker = None

    for prompt in tqdm.tqdm(medium_set):
        samples_path: Union[str, LiteralString, bytes] = os.path.join(
            checkpoint_path, 'samples',
            f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
            'version_0', prompt.format(f"{config['placeholder_token']} {config['class_name']}")
        )
        if os.path.exists(samples_path) and len(os.listdir(samples_path)) == args.num_images_per_medium_prompt:
            continue

        batch_size = args.batch_size_medium
        generator = fix_seed(prompt, args.seed, 'cuda')
        shape = (args.num_images_per_medium_prompt, pipe.unet.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
        latents = randn_tensor(shape, generator=generator, dtype=pipe.unet.dtype, device=pipe.unet.device)
        n_batches = (args.num_images_per_medium_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            images_batch = pipe(
                prompt.format(f"{config['placeholder_token']} {config['class_name']}"),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, num_images_per_prompt=batch_size,
                generator=generator,
                latents=latents[i * batch_size: (i + 1) * batch_size],
            ).images
            images += images_batch

        os.makedirs(samples_path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(samples_path, f'{idx}.png'))

    for prompt in tqdm.tqdm(base_set):
        samples_path = os.path.join(
            checkpoint_path, 'samples',
            f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
            'version_0', prompt.format(f"{config['placeholder_token']} {config['class_name']}")
        )
        if os.path.exists(samples_path) and len(os.listdir(samples_path)) == args.num_images_per_base_prompt:
            continue

        batch_size = args.batch_size_base
        generator = fix_seed(prompt, args.seed, 'cuda')
        shape = (args.num_images_per_base_prompt, pipe.unet.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
        latents = randn_tensor(shape, generator=generator, dtype=pipe.unet.dtype, device=pipe.unet.device)
        n_batches = (args.num_images_per_base_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            images_batch = pipe(
                prompt.format(f"{config['placeholder_token']} {config['class_name']}"),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, num_images_per_prompt=batch_size,
                generator=generator,
                latents=latents[i * batch_size: (i + 1) * batch_size],
            ).images
            images += images_batch

        os.makedirs(samples_path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(samples_path, f'{idx}.png'))


if __name__ == '__main__':
    main(parse_args())
