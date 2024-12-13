#!/bin/bash

global_batch_size=512
device_batch_size=16
accum_steps=$(($global_batch_size / $device_batch_size / 8))

python_cmd="finetune_bart.py
    --output_dir ~/scr/outputs/bart
    --logging_strategy steps
    --logging_steps 1000
    --evaluation_strategy steps
    --eval_steps 1000
    --save_strategy no
    --max_steps 10000
    --per_device_train_batch_size $device_batch_size
    --per_device_eval_batch_size $((2 * $device_batch_size))
    --gradient_accumulation_steps $accum_steps
    --warmup_ratio 0.1
    --learning_rate 5e-5
    --weight_decay 0.001
    --overwrite_output_dir
    --predict_with_generate
    --generation_max_length 64
    --generation_num_beams 5
    --fp16
"
deepspeed $python_cmd
