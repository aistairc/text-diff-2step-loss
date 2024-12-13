import os

dir_name = "/scratch/aae15163zd/outputs/diffusion_fin/MSNews-T-1000-mend-0.005-oend-0.1-otype-constant-loss-ctc-K-10000-lr-5e-5-freq-custom-lsteps-2-b1/checkpoint-10000/"
file_path = "preds-100.txt"
new_file_path = "preds-100-fixed.txt"

preds = open(os.path.join(dir_name, file_path)).read().splitlines()

new_preds = []
for l in preds:
    if not l:
        new_preds.append(".")
    else:
        new_preds.append(l)

with open(os.path.join(dir_name, new_file_path), "w") as f:
    f.write("\n".join(new_preds + [""]))

