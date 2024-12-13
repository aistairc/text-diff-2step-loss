import os
import pickle
import json
import random
import numpy as np
from collections import deque
from datasets import load_from_disk, Dataset, DatasetDict
import evaluate
import torch
from torch import nn

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, EvalPrediction
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from evaluator import Evaluate

import wandb
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_PROJECT"] = "Diffusion-speed"

parser = HfArgumentParser((Seq2SeqTrainingArguments))
training_args = parser.parse_args_into_dataclasses()[0]

model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

source_max_length = 512
target_max_length = training_args.generation_max_length

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.mask_token_id = tokenizer.mask_token_id


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["src"], padding="max_length", truncation=True, max_length=source_max_length)
    outputs = tokenizer(batch["trg"], padding="max_length", truncation=True, max_length=target_max_length)

    batch["input_ids"] = inputs.input_ids

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()

    return batch

data_dir = f"/scratch/aae15163zd/corpus/datasets/XSum"
dataset_dir = os.path.join(data_dir, f"bart-src-{source_max_length}-trg-{target_max_length}")
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

        batch["decoder_input_ids"] = outputs.input_ids
        batch["labels"] = outputs.input_ids.copy()

        return batch

    ds = ds.map(
        process_data_to_model_inputs,
        batched=True,
        num_proc=40,
        keep_in_memory=True
    )
    ds.save_to_disk(dataset_dir)

test_dataset = ds["test"].select(range(1000))

def compute_metrics(p: EvalPrediction):
    return {}

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
)

trainer.evaluate()
