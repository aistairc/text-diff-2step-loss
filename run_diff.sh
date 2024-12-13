#!/bin/bash


global_batch_size=512
device_batch_size=64
accum_steps=$(($global_batch_size / $device_batch_size / 8))

# XSum, Squad, MSNews, MSQG
task=Squad
loss=ctc
loss_steps=1
freq=custom
T=1000
mend=0.005
oend=0.1
otype=constant
max_steps=10000
lr=5e-5
run_name=${task}-T-${T}-mend-${mend}-oend-${oend}-otype-${otype}-loss-${loss}-K-${max_steps}-lr-${lr}-freq-${freq}-lsteps-$loss_steps
#run_name=${task}-T-${T}-mend-${mend}-oend-${oend}-otype-${otype}-loss-${loss}-K-${max_steps}-lr-${lr}-1step

python_cmd="finetune_qg.py
    --run_name $run_name
    --output_dir ~/scr/outputs/diffusion_fin/$run_name
    --logging_strategy steps
    --logging_steps 1000
    --evaluation_strategy steps
    --eval_steps 1000
    --save_strategy no
    --save_steps 1000
    --max_steps $max_steps
    --per_device_train_batch_size $device_batch_size
    --per_device_eval_batch_size $((2 * $device_batch_size))
    --gradient_accumulation_steps $accum_steps
    --lr_scheduler_type cosine
    --learning_rate $lr
    --warmup_ratio 0.1
    --weight_decay 0.001
    --overwrite_output_dir
    --fp16
    --num_diffusion_steps $T
    --num_inference_steps 20
    --num_steps_for_loss $loss_steps
    --mask_prob_end $mend
    --other_prob_end $oend
    --other_prob_scheduler_type $otype
    --loss_type $loss
    --freq_noise_drawing $freq
    --task_name $task
"

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 29600 $python_cmd
