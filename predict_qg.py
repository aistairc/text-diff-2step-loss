import os
import pickle
import json
import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datasets import load_from_disk, Dataset, DatasetDict
#import transformers
#import evaluate
import torch
from torch import nn
import logging

from nar_transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
#from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers.models.encoder_decoder.modeling_baseline import BaselineModel
from nar_transformers.models.bart.modeling_diffusion_bart import BartModel, BartForConditionalGeneration
from nar_transformers import Trainer, EvalPrediction
from nar_transformers import HfArgumentParser, TrainingArguments

from diff_args import DiffusionInferenceArguments, DiffusionInferenceArguments
from evaluator import Evaluate

import wandb
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "NAR-NAACL"
run_name = "qg-squad-small"

parser = HfArgumentParser((TrainingArguments, DiffusionInferenceArguments))
training_args, diff_infer_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(diff_infer_args.finetuned_model_path)
model = BartForConditionalGeneration.from_pretrained(diff_infer_args.finetuned_model_path)
model.init_schedule()
model.init_diff_params(diff_infer_args)

source_max_length = 512
target_max_length = 128

data_dir = f"/scratch/aae15163zd/corpus/datasets/{model.config.task_name}/"
dataset_dir = os.path.join(data_dir, f"diff-src-{source_max_length}-trg-{target_max_length}")
if os.path.exists(dataset_dir):
    ds = load_from_disk(dataset_dir, keep_in_memory=True)
else:
    d = {}
    for split in ("train", "valid", "test"):
        with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
            jsonl_data = [json.loads(l) for l in f.readlines()]
        d[split] = Dataset.from_list(jsonl_data)
    ds = DatasetDict(d)

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(batch["src"], padding="max_length", truncation=True, max_length=source_max_length)
        outputs = tokenizer(batch["trg"], padding="max_length", truncation=True, max_length=target_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        return batch

    ds = ds.map(
        process_data_to_model_inputs,
        batched=True,
        num_proc=40,
    )
    ds.save_to_disk(dataset_dir)

test_dataset = ds["test"]

import pickle
def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    #label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids = p.predictions

    # Removing repetition tokens
    pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [s.lower().strip() for s in pred_str]
    label_str = [s.lower().strip() for s in label_str]

    with open(os.path.join(diff_infer_args.finetuned_model_path, f"refs.txt"), "w") as f:
        f.write("\n".join(label_str + [""]))
    #with open(os.path.join(diff_infer_args.finetuned_model_path, f"preds-{diff_infer_args.num_inference_steps}_.txt"), "w") as f:
    #with open(os.path.join(diff_infer_args.finetuned_model_path, f"preds-{diff_infer_args.num_inference_steps}.txt"), "w") as f:
    pred_file_name = f"preds-{diff_infer_args.num_inference_steps}.txt"
    with open(os.path.join(diff_infer_args.finetuned_model_path, pred_file_name), "w") as f:
        f.write("\n".join(pred_str + [""]))

    return {}

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
)

trainer.evaluate()
