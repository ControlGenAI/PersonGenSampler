import os
import sys
import copy
import argparse

import torch.backends.cuda
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

sys.path.append('./../../')
# noinspection PyProtectedMember
from nb_utils.utils import _read_config
from persongen.utils.seed import fix_seed, get_seed
from persongen.model.svd_diff import setup_module_for_svd_diff
from baselines.profusion.pipeline import StableDiffusionPipeline, StableDiffusionPipelineOFT


def load_svd_diff(config, checkpoint_idx, device, use_original_model=False):
    scheduler = DDIMScheduler.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="scheduler"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="unet"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="vae", revision=config['revision']
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="tokenizer", revision=config['revision']
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="text_encoder", revision=config['revision']
    ).to(device)

    if use_original_model:
        original_unet = copy.deepcopy(unet) if config['finetune_unet'] else unet
        original_text_encoder = copy.deepcopy(text_encoder) if config['finetune_text_encoder'] else text_encoder

        original_unet.eval(), original_text_encoder.eval()

    checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{checkpoint_idx}')

    if config['finetune_text_encoder']:
        setup_module_for_svd_diff(
            text_encoder, scale=1.0, qkv_only=config['qkv_only'],
            deltas_path=os.path.join(checkpoint_path, 'text_encoder.bin')
        )
    if config['finetune_unet']:
        setup_module_for_svd_diff(
            unet, scale=1.0, qkv_only=config['qkv_only'],
            deltas_path=os.path.join(checkpoint_path, 'unet.bin')
        )

    unet.eval(), vae.eval(), text_encoder.eval()

    if use_original_model:
        # noinspection PyUnboundLocalVariable
        return scheduler, (unet, original_unet), vae, tokenizer, (text_encoder, original_text_encoder)

    return scheduler, unet, vae, tokenizer, text_encoder


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
        "--use_original_model",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--use_empty_ref_prompt",
        action="store_true",
        default=False
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
        default=0
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

    scheduler, unet, vae, tokenizer, text_encoder = load_svd_diff(
        config, args.checkpoint_idx, device, args.use_original_model
    )

    # noinspection PyTypeChecker
    if args.use_original_model:
        (unet, original_unet) = unet
        (text_encoder, original_text_encoder) = text_encoder

        pipe = StableDiffusionPipelineOFT(
            vae=vae,
            tokenizer=tokenizer,
            unet=unet,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            original_unet=original_unet,
            original_text_encoder=original_text_encoder
        ).to(device)
    else:
        pipe = StableDiffusionPipeline(
            vae=vae,
            tokenizer=tokenizer,
            unet=unet,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)

    n_batches = (args.num_images_per_prompt - 1) // args.batch_size + 1
    for seed_prompt in args.prompts.split('#'):
        prompt = seed_prompt.format(f'{config["placeholder_token"]} {config["class_name"]}')
        if args.use_empty_ref_prompt:
            ref_prompt = seed_prompt.format('')
        else:
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
            ), generator=generator, dtype=torch.float32, device=device
        )

        if args.refine_step > 0:
            generator_noise = torch.Generator(device=device)
            generator_noise = generator_noise.manual_seed(get_seed(seed_prompt, seed=args.seed + args.refine_seed))
            noise = randn_tensor(
                (args.refine_step, args.num_inference_steps, *latents.shape), generator=generator_noise, dtype=torch.float32, device=device
            )
        else:
            noise = None

        inference_folder_name = (
            f"ns{args.num_inference_steps}_gs{args.guidance_scale}_sg{args.guidance_scale_ref}"
            f"_rs{args.refine_step}_re{args.refine_eta}_rgs{args.refine_guidance_scale}"
        )
        if args.use_empty_ref_prompt:
            inference_folder_name = inference_folder_name + '_ep'
        if args.use_original_model:
            inference_folder_name = inference_folder_name + '_noft'
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
