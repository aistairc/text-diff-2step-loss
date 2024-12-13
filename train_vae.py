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
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer, EvalPrediction

from nar_transformers.models.encoder_decoder.modeling_vae import VAE

import wandb
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "NAR-NAACL"
run_name = "qg-squad-small"

batch_size = 20  # change to 16 for full training
source_max_length = 512 # 128 actually works better for MT
target_max_length = 64 # 128 actually works better for MT
grad_accum_steps = 1
linesep_token = "<lsep>"


#model_name = "prajjwal1/bert-small"
model_name = "prajjwal1/bert-tiny"
#model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VAE.from_encoder_decoder_pretrained(model_name, model_name)
#model.load_state_dict(torch.load("model.pt"))
#model = VAE.from_pretrained("/groups/gac50543/migrated_from_SFA_GPFS/asada/outputs/nar/ae/checkpoint-3000/")

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

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    #output_dir="/groups/gac50543/migrated_from_SFA_GPFS/asada/outputs/nar/ae",
    output_dir="/groups/gac50543/migrated_from_SFA_GPFS/asada/outputs/nar/foo",
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    max_steps=1000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    warmup_ratio=0.1,
    learning_rate=1e-04,
    weight_decay=0.1,
    overwrite_output_dir=True,
    save_total_limit=False,
    fp16=True,
    #torch_compile=True,
    report_to="wandb",
    run_name=run_name,
)

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
torch.save(model.state_dict(), "model-emb-vae-dim8.pt")
#trainer.train(resume_from_checkpoint=True)
