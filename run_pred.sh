#!/bin/bash

export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export HF_HOME="/scratch/aae15163zd/cache/huggingface"

python_cmd="predict_qg.py
    --output_dir ~/scr/outputs/diffusion/for_eval
    --per_device_eval_batch_size 32
    --max_steps 1
    --num_inference_steps 300
    --finetuned_model_path /scratch/aae15163zd/outputs/diffusion_fin/XSum-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-80000-lr-5e-5-freq-custom-lsteps-2/checkpoint-80000
"

deepspeed $python_cmd
