data_files = {
  "train": "C:/Users/sabar/data/twitter_entity_sentiment/train.txt",
  "validation": "C:/sabar/YourUser/data/twitter_entity_sentiment/valid.txt"
}
output_dir = "C:/Users/sabar/models/cased-bert-dapt"
model_name = "bert-base-cased"
block_size = 512
# pretrain_cased_bert.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import math

data_files = {
    "train": r"C:\Users\sabar\OneDrive\Desktop\archive\train.txt",
    "validation": r"C:\Users\sabar\OneDrive\Desktop\archive\valid.txt"
}
output_dir = r"C:\Users\sabar\OneDrive\Desktop\archive\cased-bert-dapt"
model_name = "bert-base-cased"
block_size = 512

# 1) load raw text dataset (each line = one example)
dataset = load_dataset("text", data_files=data_files)

# 2) tokenizer + model (masked LM head)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 3) tokenize examples (keeps special token masks for MLM collator)
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4) concatenate and split into fixed-length blocks (BERT's input blocks)
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        total_length = 0
    result = {
        k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    return result

lm_datasets = tokenized.map(group_texts, batched=True)

# 5) data collator that prepares masked inputs for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 6) training arguments — tune these for your hardware/dataset
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=0.1,                 # start with 1, you can increase later
    per_device_train_batch_size=12,     # ← try 12 first (good starting point for 6 GB)
    per_device_eval_batch_size=12,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,                 # keep only last 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,                   # more frequent logging
    logging_dir=f"{output_dir}/logs",
    fp16=True,                          # ← this enables mixed precision → big speed + lower memory
    # gradient_accumulation_steps=2,    # optional – uncomment if you get OOM and want bigger effective batch
    report_to="none", 
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets.get("validation"),
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
