import sys
import argparse

from persongen.trainer import trainers

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Config args
    parser.add_argument("--project_name", type=str, default='persongen')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)

    # Training args
    parser.add_argument("--trainer_type", type=str, required=True, choices=list(trainers.classes.keys()))
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--num_train_epochs", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_val_imgs", type=int, default=5)

    # Data args
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, default=None, required=True)
    parser.add_argument("--train_data_dir", type=str, default=None, required=True)

    # Model args
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--qkv_only", action="store_true", default=False)
    parser.add_argument("--finetune_transformer", action='store_true', default=True)
    parser.add_argument("--finetune_text_encoder", action='store_true', default=False)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stable-diffusion-2-base')

    # Prior preservation args
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--class_data_dir", type=str, default=None, help="A folder containing the training data of class images.")
    parser.add_argument("--with_prior_preservation", action="store_true", default=False, help="Flag to add prior preservation loss.")

    # Optimizer args
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--learning_rate_1d", type=float, default=None)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")

    args = parser.parse_args()
    args.argv = [sys.executable] + sys.argv

    return args


def main(args):
    trainer = trainers[args.trainer_type](args)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main(parse_args())
