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
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_PROJECT"] = "Diffusion-speed"

parser = HfArgumentParser((TrainingArguments, DiffusionInferenceArguments))
training_args, diff_infer_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(diff_infer_args.finetuned_model_path)
model = BartForConditionalGeneration.from_pretrained(diff_infer_args.finetuned_model_path)
model.init_schedule()
model.init_diff_params(diff_infer_args)

source_max_length = 512
target_max_length = training_args.max_steps

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
        keep_in_memory=True,
    )
    ds.save_to_disk(dataset_dir)

test_dataset = ds["test"].select(range(1000))

import pickle
def compute_metrics(p: EvalPrediction):
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
