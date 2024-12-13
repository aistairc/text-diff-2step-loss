import os
import pickle
import random
import numpy as np
from collections import deque
from datasets import load_from_disk
#import transformers
import evaluate
import torch
from torch import nn

from nar_transformers import AutoTokenizer, AutoModelForCausalLM
#from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers.models.encoder_decoder.modeling_baseline import BaselineModel
from nar_transformers import Trainer, EvalPrediction
from nar_transformers import HfArgumentParser, TrainingArguments

import wandb
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "NAR-NAACL"
run_name = "qg-squad-small"

parser = HfArgumentParser((TrainingArguments))
training_args = parser.parse_args_into_dataclasses()[0]

#model_name = "google-bert/bert-large-uncased"
#model_name = "google-bert/bert-base-uncased"
#model_name = "google/bert_uncased_L-8_H-512_A-8" # medium
model_name = "google/bert_uncased_L-4_H-512_A-8" # small
#model_name = "google/bert_uncased_L-2_H-128_A-2" # tiny
#model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
model = BaselineModel.from_encoder_decoder_pretrained(model_name, model_name)

source_max_length = 256
target_max_length = 256

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.mask_token_id = tokenizer.mask_token_id

# We use the same data splitting as Du et al. 2017
datasets = load_from_disk("/groups/gac50543/migrated_from_SFA_GPFS/asada/corpus/squad-du-split/squad-hf-dd/")

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["answer"], batch["context"], padding="max_length", truncation="only_second",
        max_length=source_max_length)
    outputs = tokenizer(batch["question"], padding="max_length", truncation=True, max_length=target_max_length)

    null_inputs = tokenizer([""] * len(batch["answer"]), padding="max_length", max_length=source_max_length)
    batch["null_input_ids"] = null_inputs.input_ids
    batch["null_attention_mask"] = null_inputs.attention_mask

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

datasets = datasets.map(
    process_data_to_model_inputs,
    batched=True,
    num_proc=40,
)

train_data = datasets["train"]
val_data = datasets["validation"]


# load bleu for validation
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
#meteor = evaluate.load("meteor")
unused_tokens = [f"[unused{i}]" for i in range(1000)]
tokenizer.add_special_tokens({"additional_special_tokens": unused_tokens})
special_token_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id,
    tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id}

import pickle
def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids = p.predictions

    #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)

    """
    pred_ids = [
        [xx for xx in x if xx not in special_token_ids] for x in pred_ids
    ]
    no_rep_pred_str = tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)
    print(no_rep_pred_str[:10])
    """

    # Removing repetition tokens
    pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print(pred_str[:3])
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    try:
        bleu4_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    except:
        bleu4_output = {"bleu": 0.0}
    #meteor_output = meteor.compute(predictions=pred_str, references=label_str)

    return {
        "bleu4": round(np.mean(bleu4_output["bleu"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
        #"meteor": round(np.mean(meteor_output["meteor"]), 4),
    }

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
torch.save(model.state_dict(), "/scratch/aae15163zd/outputs/nar/ctc-baseline-token-crossatt.pt")
#trainer.train(resume_from_checkpoint=True)
