# Inference

We provide CLI for different sampling methods as well as a [notebook example](<../Training & Inference.ipynb>). 

## Weighted combination of trajectories

You can use this code for inference of a SVDDiff/Dreambooth model over PixArt. The only difference is that SVDDiff models require `svd_` prefix before the `--inference_type` argument. Here is a complete list of different samplings used in the paper:

### Base
```bash
python ./inference.py \
    --inference_type="base" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=20 \
    
    --guidance_scale=4.5 \             # Corresponds to w
    
    --version=0 \
    --replace_inference_output 
```

### Switching
```bash
python ./inference.py \    
    --inference_type="multistage" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=20 \
    
    --guidance_scale=4.5 \             # Corresponds to w
    --guidance_scale_ref=0.0 \
    --change_step=8 \                  # Corresponds to t_{sw}
 
    --version=0 \
    --replace_inference_output 
```

### Mixed
```bash
python ./inference.py \
    --inference_type="multistage" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=20 \

    --guidance_scale=2.5 \
    --guidance_scale_ref=2.0 \
    --change_step=-1 \

    --version=0 \
    --replace_inference_output 
```
