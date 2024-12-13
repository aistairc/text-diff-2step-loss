import os
import pickle
import json
import random
import numpy as np
from collections import deque
from datasets import load_from_disk, Dataset, DatasetDict
#import transformers
#import evaluate
import torch
from torch import nn

from nar_transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from nar_transformers.models.bart.modeling_diffusion_bart import BartModel, BartForConditionalGeneration
from nar_transformers import Trainer, EvalPrediction
from nar_transformers import HfArgumentParser, TrainingArguments

from diff_args import DiffusionTrainingArguments, DiffusionInferenceArguments
from evaluator import Evaluate

import wandb
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_PROJECT"] = "Diffusion-loss"

parser = HfArgumentParser((TrainingArguments, DiffusionTrainingArguments, DiffusionInferenceArguments))
training_args, diff_args, diff_infer_args = parser.parse_args_into_dataclasses()
print(diff_args, diff_infer_args)

#model_name = "google-bert/bert-large-uncased"
#model_name = "google-bert/bert-base-uncased"
#model_name = "google/bert_uncased_L-8_H-512_A-8" # medium
#model_name = "google/bert_uncased_L-4_H-512_A-8" # small
#model_name = "google/bert_uncased_L-2_H-128_A-2" # tiny
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(["[BLANK]"], special_tokens=True)
#model = BaselineModel.from_encoder_decoder_pretrained(model_name, model_name, tokenizer)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Add blank token
model.resize_token_embeddings(len(tokenizer))
model.config.blank_token_id = tokenizer.convert_tokens_to_ids(["[BLANK]"])[0]

source_max_length = 512
target_max_length = 128

model.init_diff_params(diff_args)
model.init_schedule()
model.init_diff_params(diff_infer_args)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.mask_token_id = tokenizer.mask_token_id

#data_dir = f"/scratch/aae15163zd/corpus/datasets/{model.config.task_name}/"
data_dir = "/home/ubuntu/asada-data/dataset/Squad/"
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
        #inputs = tokenizer(batch["answer"], batch["context"], padding="max_length", truncation="only_second",
        #    max_length=source_max_length)
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

calculator = Evaluate()

import pickle
def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    pred_ids = p.predictions

    # Removing repetition tokens
    pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print(pred_str[:3])
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [s.lower().strip() for s in pred_str]
    label_str = [s.lower().strip() for s in label_str]

    eval_dict = calculator.evaluate(pred_str, label_str)
    # Remove items for ease of viewing
    for i in (1, 2, 3):
        eval_dict.pop(f"Bleu_{i}")

    return eval_dict

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

trainer.train()
