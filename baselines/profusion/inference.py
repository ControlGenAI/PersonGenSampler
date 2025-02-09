import os
import sys
import argparse

import torch.backends.cuda
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

sys.path.append('./../../')
# noinspection PyProtectedMember
from nb_utils.utils import _read_config
from persongen.utils.seed import fix_seed, get_seed
from persongen.model.svd_diff import setup_module_for_svd_diff
from baselines.profusion.pipeline import StableDiffusionPipeline

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def load_svd_diff(config, checkpoint_idx, device, dtype):
    scheduler = DDIMScheduler.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="scheduler", revision=config['revision']
    )
    unet = UNet2DConditionModel.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="unet", revision=config['revision']
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix', torch_dtype=dtype
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="tokenizer", revision=config['revision']
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="tokenizer_2", revision=config['revision']
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="text_encoder", revision=config['revision']
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config['pretrained_model_name_or_path'], torch_dtype=dtype,
        subfolder="text_encoder_2", revision=config['revision']
    ).to(device)

    checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{checkpoint_idx}')

    if config['finetune_text_encoder']:
        setup_module_for_svd_diff(
            text_encoder, scale=1.0, qkv_only=config['qkv_only'],
            deltas_path=os.path.join(checkpoint_path, 'text_encoder.bin'), fuse=True
        )
        text_encoder.to(dtype)

        setup_module_for_svd_diff(
            text_encoder_2, scale=1.0, qkv_only=config['qkv_only'],
            deltas_path=os.path.join(checkpoint_path, 'text_encoder_2.bin'), fuse=True
        )
        text_encoder_2.to(dtype)

    if config['finetune_unet']:
        setup_module_for_svd_diff(
            unet, scale=1.0, qkv_only=config['qkv_only'],
            deltas_path=os.path.join(checkpoint_path, 'unet.bin'), fuse=True
        )
        unet.to(dtype)

    unet.eval(), vae.eval(), text_encoder.eval(), text_encoder_2.eval()

    return scheduler, unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2


def sampling_kwargs(
        prompt, ref_prompt=None,
        num_inference_steps=50, guidance_scale=5.0, guidance_scale_ref=5.0,
        refine_step=0, refine_eta=1., refine_guidance_scale=5.0
):
    kwargs = {
        "prompt": prompt,
        "ref_prompt": ref_prompt,

        "guidance_scale": guidance_scale,
        "guidance_scale_ref": guidance_scale_ref,
        "num_inference_steps": num_inference_steps,

        "refine_step": refine_step,
        "refine_eta": refine_eta,
        "refine_guidance_scale": refine_guidance_scale
    }

    return kwargs


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--prompts",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=30,
        help="Number of generated images for each base prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--guidance_scale_ref",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )

    parser.add_argument(
        "--refine_step",
        type=int,
        default=0
    )
    parser.add_argument(
        "--refine_eta",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--refine_guidance_scale",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )
    parser.add_argument(
        "--refine_seed",
        type=int,
        default=10
    )

    parser.add_argument(
        "--version",
        type=int,
        default=0
    )

    return parser.parse_args()


def main(args):
    # Change path_mapping if necessary
    path_mapping = {
        '<Path to the folder where all checkpoints were trained>':
            '<Path to the folder where all checkpoints are located>'
    }
    config = _read_config(args.config_path, path_mapping=path_mapping)

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    scheduler, unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2 = load_svd_diff(config, args.checkpoint_idx, device, torch.float16)
    # noinspection PyTypeChecker
    pipe = StableDiffusionPipeline(
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        scheduler=scheduler,
        feature_extractor=None
    ).to(device)
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    n_batches = (args.num_images_per_prompt - 1) // args.batch_size + 1
    for seed_prompt in args.prompts.split('#'):
        prompt = seed_prompt.format(f'{config["placeholder_token"]} {config["class_name"]}')
        ref_prompt = seed_prompt.format(config["class_name"])

        pipe_kwargs = sampling_kwargs(
            prompt=args.batch_size * [prompt],
            ref_prompt=args.batch_size * [ref_prompt],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale, guidance_scale_ref=args.guidance_scale_ref,
            refine_step=args.refine_step, refine_eta=args.refine_eta, refine_guidance_scale=args.refine_guidance_scale
        )

        generator = fix_seed(seed_prompt, s=args.seed, device=device)
        latents = randn_tensor(
            (
                args.num_images_per_prompt, pipe.unet.config.in_channels,
                pipe.unet.config.sample_size, pipe.unet.config.sample_size
            ), generator=generator, dtype=pipe.unet.dtype, device=device
        )

        if args.refine_step > 0:
            generator_noise = torch.Generator(device=device)
            generator_noise = generator_noise.manual_seed(get_seed(seed_prompt, seed=args.seed + args.refine_seed))
            noise = randn_tensor(
                (args.refine_step, args.num_inference_steps, *latents.shape),
                generator=generator_noise, dtype=pipe.unet.dtype, device=device
            )
        else:
            noise = None

        inference_folder_name = (
            f"ns{args.num_inference_steps}_gs{args.guidance_scale}_sg{args.guidance_scale_ref}"
            f"_rs{args.refine_step}_re{args.refine_eta}_rgs{args.refine_guidance_scale}"
        )
        samples_path = os.path.join(
            config['output_dir'], f'checkpoint-{args.checkpoint_idx}',
            'samples', inference_folder_name, f'version_{args.version}'
        )
        path = os.path.join(samples_path, prompt)
        os.makedirs(path, exist_ok=True)

        if len(os.listdir(path)) >= args.num_images_per_prompt:
            print(f'Skip prompt {prompt}')
            continue

        images = []
        for idx in range(n_batches):
            images_batch = pipe(
                latents=latents[idx * args.batch_size:(idx + 1) * args.batch_size],
                refine_noise=noise[:, :, idx * args.batch_size:(idx + 1) * args.batch_size] if noise is not None else None,
                generator=generator,
                **pipe_kwargs
            ).images
            images += images_batch

        for idx, image in enumerate(images):
            image.save(os.path.join(path, f'{idx}.png'))


if __name__ == "__main__":
    main(parse_args())
