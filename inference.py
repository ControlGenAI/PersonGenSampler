import argparse

import yaml

from persongen.inferencer import inferencers
from nb_utils.configs import live_object_data
from nb_utils.eval_sets import evaluation_sets

import warnings

warnings.filterwarnings('ignore')


def parse_args():
    # Inferencer type
    type_parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    type_parser.add_argument("--inference_type", type=str, required=True, choices=list(inferencers.classes.keys()))
    type_args, _ = type_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[type_parser], add_help=False)
    # Inference args
    parser.add_argument("--batch_size_base", type=int, default=10)
    parser.add_argument("--batch_size_medium", type=int, default=10)
    parser.add_argument("--num_images_per_base_prompt", type=int, default=30)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    # Text condition args
    parser.add_argument("--with_class_name", default=False, action="store_true")

    # Model args
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_idx", type=str, default=None)

    # Save args
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--replace_inference_output", action='store_true', default=False)
    parser.add_argument("--output_dir", type=str, default=None, help='Override output dir in the config')

    # Base Sampler args
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)

    # SVD args
    if 'svd' in type_args.inference_type:
        parser.add_argument("--svd_scale", type=float, default=1.0)
    # Multistage args
    if 'multistage' in type_args.inference_type:
        parser.add_argument("--change_step", type=int, required=True)
        parser.add_argument("--guidance_scale_ref", type=float, required=True)
    # Photoswap args
    if 'photoswap' in type_args.inference_type:
        parser.add_argument("--guidance_scale_ref", type=float, required=True)
        parser.add_argument("--photoswap_sf_step", type=int, required=True)
        parser.add_argument("--photoswap_cm_step", type=int, required=True)
        parser.add_argument("--photoswap_sm_step", type=int, required=True)
    # Cross Attention Masked args
    if 'crossattn_masked' in type_args.inference_type:
        parser.add_argument("--change_step", type=int, required=True)
        parser.add_argument("--inner_gs_1", type=float, required=True)
        parser.add_argument("--inner_gs_2", type=float, required=True)
        parser.add_argument("--out_gs_1", type=float, required=True)
        parser.add_argument("--out_gs_2", type=float, required=True)
        parser.add_argument("--quantile", type=float, required=True)

    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    evaluation_set = evaluation_sets[live_object_data[config['class_name']]]
    inferencer = inferencers[args.inference_type](config, args, evaluation_set, evaluation_sets['base'])

    inferencer.setup()
    inferencer.generate()


if __name__ == '__main__':
    main(parse_args())
