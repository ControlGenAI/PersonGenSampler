# Training

For training, you need to define a dataset (`--test_data_dir`) and preprocessed training images (`--instance_data_dir` or `--train_data_dir`) and then run the CLI for training.

We use [Weights & Biases](https://wandb.ai/home) to track experiments. Before training, you should put your W&B API key into the `WANDB_KEY` environment variable or pass it using `--api_key` argument.

Our source code allows the SVDDiff training over SDXL that is used in the main experiments. Here is an example how to train the model for `dog6` concept from Dreambooth dataset.

We tested our code only in a single GPU setup on Nvidia V100/A100. It should take less than an hour to train a single model.

## SVDDiff

```bash
XFORMERS_DISABLED=True python ./train.py \
  --test_data_dir="./dreambooth/dataset/dog6" \
  --train_data_dir="./dreambooth/aug_dataset/dog6" \
  --class_name="dog" \
  --output_dir="./training-runs/" \
  --mixed_precision="no" \
  --trainer_type="svd" \
  --train_batch_size=1 \
  --resolution=1024 \
  --num_train_epochs=2000 \
  --checkpointing_steps=200 \
  --num_val_imgs=5 \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --placeholder_token="sks" \
  --mixed_precision='fp16' \
  --qkv_only \
  --finetune_unet \
  --learning_rate=0.001 \
  --learning_rate_1d=1e-6
```
