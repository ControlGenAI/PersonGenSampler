# Training

For training, you need to define a dataset (`--test_data_dir`) and preprocessed training images (`--instance_data_dir` or `--train_data_dir`) and then run the CLI for training.

We use [Weights & Biases](https://wandb.ai/home) to track experiments. Before training, you should put your W&B API key into the `WANDB_KEY` environment variable or pass it using `--api_key` argument.

Our source code allows the training of multiple baselines as well as SVDDiff that is used in the main experiments. Here is an example how to train all models for `dog6` concept from Dreambooth dataset.

We tested our code only in a single GPU setup on Nvidia A100/V100. It should take less than an hour to train any single model.

## Custom Diffusion

```bash
python ./baselines/custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --test_data_dir="./dreambooth/dataset/dog6" \
  --instance_data_dir="./dreambooth/aug_dataset/dog6" \
  --output_dir="./baselines/custom_diffusion/training-runs/" \
  --instance_prompt="a photo of a sks dog" \
  --modifier_token "sks" \
  --class_name="dog" \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1600 \
  --checkpointing_steps=200 \
  --scale_lr
 ```

## Textual Inversion

```bash
python ./baselines/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base"  \
  --test_data_dir="$./dreambooth/dataset/dog6" \
  --train_data_dir="./dreambooth/aug_dataset/dog6" \
  --output_dir="./baselines/textual_inversion/training-runs/" \
  --learnable_property="object" \
  --placeholder_token="<dog>" --initializer_token="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5.0e-03 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 
 ```

## Dreambooth

```bash
python ./train.py \
  --test_data_dir="./dreambooth/dataset/dog6" \
  --train_data_dir="./dreambooth/aug_dataset/dog6" \
  --class_name="dog" \
  --output_dir="./training-runs/" \
  --mixed_precision="no" \
  --trainer_type="base" \
  --train_batch_size=1 \
  --num_train_epochs=2000 \
  --checkpointing_steps=200 \
  --num_val_imgs=5 \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --placeholder_token="sks" \
  --qkv_only \
  --finetune_unet \
  --finetune_text_encoder \
  --learning_rate=2e-5
```

## SVDDiff

```bash
python ./train.py \
  --test_data_dir="./dreambooth/dataset/dog6" \
  --train_data_dir="./dreambooth/aug_dataset/dog6" \
  --class_name="dog" \
  --output_dir="./training-runs/" \
  --mixed_precision="no" \
  --trainer_type="svd" \
  --train_batch_size=1 \
  --num_train_epochs=2000 \
  --checkpointing_steps=200 \
  --num_val_imgs=5 \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --placeholder_token="sks" \
  --finetune_unet \
  --finetune_text_encoder \
  --learning_rate=0.001 \
  --learning_rate_1d=1e-6
```
