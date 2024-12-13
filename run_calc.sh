#!/bin/bash

#output_dir="/scratch/aae15163zd/outputs/diffusion/Squad-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ctc-K-10000-lr-5e-5-freq-custom-lsteps-2/checkpoint-4000"
#output_dir="/scratch/aae15163zd/outputs/diffusion/XSum-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-40000-lr-5e-5-freq-custom-lsteps-2/checkpoint-15000"
#output_dir="/scratch/aae15163zd/outputs/diffusion/MSQG-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-5000-lr-1e-4/checkpoint-5000"
#output_dir="/scratch/aae15163zd/outputs/diffusion/MSNews-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-10000-lr-5e-5-freq-custom-lsteps-1/checkpoint-10000"
#output_dir="/scratch/aae15163zd/outputs/diffusion_fin/Squad-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ctc-K-10000-lr-5e-5-freq-custom-lsteps-2/checkpoint-4000"
#output_dir="/scratch/aae15163zd/outputs/diffusion_fin/MSQG-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ctc-K-10000-lr-5e-5-freq-custom-lsteps-2/checkpoint-4000"
output_dir="/scratch/aae15163zd/outputs/diffusion_fin/MSNews-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-10000-lr-5e-5-freq-custom-lsteps-2-b1/checkpoint-10000"
#output_dir="/scratch/aae15163zd/outputs/diffusion_fin/XSum-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ce-K-80000-lr-5e-5-freq-custom-lsteps-2/checkpoint-80000"
#output_dir="/scratch/aae15163zd/outputs/diffusion_fin/XSum-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ctc-K-40000-lr-5e-5-freq-custom-lsteps-2/checkpoint-28000"
python calc_all.py \
    --ref_path ${output_dir}/refs.txt \
    --pred_path ${output_dir}/preds-100.txt
