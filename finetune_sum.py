import os
import pickle
import json
import random
import numpy as np
from collections import deque
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
#import transformers
import evaluate
from torch import nn

from nar_transformers import AutoTokenizer, AutoModelForCausalLM
#from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers.models.encoder_decoder.modeling_baseline import BaselineModel
from nar_transformers.models.bart.modeling_diffusion_bart import BartModel, BartForConditionalGeneration
from nar_transformers import Trainer, EvalPrediction
from nar_transformers import HfArgumentParser, TrainingArguments

from diff_args import DiffusionArguments

import wandb
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "NAR-NAACL"
run_name = "qg-squad-small"

parser = HfArgumentParser((TrainingArguments, DiffusionArguments))
training_args, diff_args = parser.parse_args_into_dataclasses()
print(training_args, diff_args)

#model_name = "google-bert/bert-large-uncased"
#model_name = "google-bert/bert-base-uncased"
#model_name = "google/bert_uncased_L-8_H-512_A-8" # medium
#model_name = "google/bert_uncased_L-4_H-512_A-8" # small
#model_name = "google/bert_uncased_L-2_H-128_A-2" # tiny
#model_name = "bert-base-uncased"
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

model.init_noise(diff_args, msl=target_max_length)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.mask_token_id = tokenizer.mask_token_id
#special_token_ids = []
#for k, v in tokenizer.special_tokens_map.items():
#    if k == "mask_token": continue
#    special_token_ids.append(tokenizer.convert_tokens_to_ids(v))
#model.config.special_token_ids = special_token_ids

src_column = "src"
tgt_column = "trg"

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch[src_column], padding="max_length", truncation=True, max_length=source_max_length)
    outputs = tokenizer(batch[tgt_column], padding="max_length", truncation=True, max_length=target_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

d_dir = "/groups/gac50543/migrated_from_SFA_GPFS/asada/corpus/datasets/XSum/"
d = {}
for split in ("train", "valid", "test"):
    with open(os.path.join(d_dir, f"{split}.jsonl")) as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]
    d[split] = Dataset.from_list(jsonl_data)
ds = DatasetDict(d)

ds = ds.map(
    process_data_to_model_inputs,
    batched=True,
    num_proc=40,
    keep_in_memory=True,
)


# load rouge for validation
rouge = evaluate.load("rouge")

import pickle
def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids = p.predictions

    #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    #print(pred_str[:3])

    # Removing repetition tokens
    pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print(pred_str[:3])
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [s.lower() for s in pred_str]
    label_str = [s.lower() for s in label_str]
    with open("./output_texts/xsum_refs.txt", "w") as f:
        f.write("\n".join(label_str + [""]))
    with open("./output_texts/xsum_preds.txt", "w") as f:
        f.write("\n".join(pred_str + [""]))

    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return {
        "rouge1": round(np.mean(rouge_output["rouge1"]), 4),
        "rouge2": round(np.mean(rouge_output["rouge2"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
    }

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    #eval_dataset=ds["valid"],
    eval_dataset=ds["test"],
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
