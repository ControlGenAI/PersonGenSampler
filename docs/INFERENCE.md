# Inference

We provide CLI for different sampling methods as well as a [notebook example](<../Training & Inference.ipynb>). 

## Weighted combination of trajectories

You can use the same code for inference CD/TI/Dreambooth/SVDDiff models. The only difference is that SVDDiff models require `svd_` prefix before the `--inference_type` argument. Here is a complete list of different samplings used in the paper:

### Base
```bash
python ./inference.py \
    --inference_type="svd_base" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \
    
    --guidance_scale=7.0 \             # Corresponds to w
    
    --version=0 \
    --replace_inference_output 
```

### Switching
```bash
python ./inference.py \    
    --inference_type="svd_multistage" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \
    
    --guidance_scale=7.0 \             # Corresponds to w
    --guidance_scale_ref=0.0 \
    --change_step=7 \                  # Corresponds to t_{sw}
 
    --version=0 \
    --replace_inference_output 
```

### Multi-Stage
```bash
python ./inference.py \
    --inference_type="svd_multistage" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \

    --guidance_scale=4.0 \             # Corresponds to w_{c}
    --guidance_scale_ref=3.0 \         # Corresponds to w_{s}
    --change_step=10 \                 # Corresponds to t_{sw}

    --version=0 \
    --replace_inference_output 
```

### Masked
```bash
python ./inference.py \
    --inference_type="svd_crossattn_masked" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \
    
    --guidance_scale=7.0 \             # Corresponds to w_{c} + w_{s} in equation (10)
    --change_step=3 \                  # Corresponds to t_{sw}
    --inner_gs_1=3.5 \                 # Corresponds to w_{c}
    --inner_gs_2=3.5 \                 # Corresponds to w_{s}
    --out_gs_1=0.0 \                   # Corresponds to w^{0}_{c}
    --out_gs_2=7.0 \                   # Corresponds to w^{0}_{s}
    --quantile=0.7 \                   # Corresponds to q

    --version=0 \
    --replace_inference_output 
```

### Photoswap
```bash
python ./inference.py \
    --inference_type="svd_photoswap" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \
    
    --guidance_scale=7.0 \
    --guidance_scale_ref=7.0 \
    --photoswap_sf_step=5 \
    --photoswap_cm_step=15 \
    --photoswap_sm_step=20 \

    --version=0 \
    --replace_inference_output 
```

### Mixed
```bash
python ./inference.py \
    --inference_type="svd_multistage" \
    --batch_size_base=10 \
    --batch_size_medium=10 \
    --num_images_per_base_prompt=10 \
    --num_images_per_medium_prompt=10 \
    --seed=0 \
    --with_class_name \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/logs/hparams.yml" \
    --checkpoint_idx=1600 \
    --num_inference_steps=50 \

    --guidance_scale=3.5 \
    --guidance_scale_ref=3.5 \
    --change_step=-1 \

    --version=0 \
    --replace_inference_output 
```


## Profusion

Our framework supports Profusion inference applied on top of the SVDDiff models:

```base
python ./baselines/profusion/inference.py \
    --config_path=<PATH_TO_THE_EXP_FOLDER>
    --checkpoint_idx=<CHECKPOINT_IDX>
    --prompts=<#_SEPARATED_LIST_OF_PROMPTS>
    --num_images_per_prompt=<NUM_IMAGES_PER_PROMPT>
    --batch_size=<BATCH_SIZE>
    --use_original_model=<ENABLE_NoFT_MODE>
    --use_empty_ref_prompt=<ENABLE_Empty_Prompt_MODE>
    --guidance_scale=<GUIDANCE_SCALE_Wc>
    --guidance_scale_ref=<GUIDANCE_SCALE_Ws>
    --num_inference_steps=<NUM_INFERENCE_STEPS>
    --refine_step=<NUMBER_OF_REFINE_STEPS>
    --refine_eta=<REFINE_ETA_r>
    --refine_guidance_scale=<REFINE_GUIDANCE_SCALE>
    --seed=<SEED_FOR_INITIAL_LATENTS>
    --refine_seed=<SEED_FOR_REFINE_NOISE>
    --version=<VERSION>
```

Example (Mixed sampling with as single Profusion refine step):
```bash
python ./baselines/profusion/inference.py \
    --config_path="./training-runs/00002-3ba5-can_SVDDiff/" \
    --checkpoint_idx=1600 \
    --prompts="a {0} in the jungle#a {0} in the snow#a {0} on the beach" \
    --num_images_per_prompt=10 \
    --batch_size=10 \
    --guidance_scale=3.5 \
    --guidance_scale_ref=3.5 \
    --num_inference_steps=50 \
    --refine_step=1 \
    --refine_eta=1.0 \
    --refine_guidance_scale=7.0 \
    --seed=0 \
    --refine_seed=10 \
    --version=0
```

## ELITE

We can perform Base and Mixed sampling from the pre-trained ELITE model:
```bash
python ./baselines/elite/inference.py \
    --global_mapper_path="./baselines/elite/checkpoints/global_mapper.pt" \
    --local_mapper_path="./baselines/elite/checkpoints/local_mapper.pt" \
    --output_dir="./baselines/elite/training-runs/"
    --placeholder_token="S" \
    --template="a {0} in the jungle#a {0} in the snow#a {0} on the beach" \
    --test_data_dir="./baselines/elite/dreambooth/backpack" \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --num_images_per_prompt=10 \
    --num_inference_steps=50 \
    --guidance_scale=3.5 \
    --guidance_scale_ref=3.5 \
    --change_step=-1 \
    --llambda=0.8 \
    --seed=0 \
    --create_exp_dir
```

You can run only Global Mapping by excluding `--local_mapper_path` argument.
