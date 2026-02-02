from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

# === Configuration ===
model_name_or_path = "C:/Users/sabar/OneDrive/Desktop/archive/cased-bert-dapt"   # your DAPT model
data_files = {
    "train": "C:/Users/sabar/OneDrive/Desktop/archive/twitter_training.csv",
    "validation": "C:/Users/sabar/OneDrive/Desktop/archive/twitter_validation.csv"
}
output_dir = "C:/Users/sabar/models/cased-bert-finetuned-sentiment"
batch_size = 16
num_train_epochs = 3
learning_rate = 3e-5

# Label mapping (adjust if your sentiments differ)
label_map = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
num_labels = 4

# 1) Load CSV (handle no header or bad lines)
dataset = load_dataset(
    "csv",
    data_files=data_files,
    column_names=["tweet_id", "entity", "sentiment", "text"],  # assume no header
    delimiter=",",  # standard CSV
    encoding="utf-8",
    on_bad_lines="skip",  # skip any malformed rows
)

# 2) Map sentiment string → integer + combine entity + text
def encode_labels(examples):
    examples["label"] = [label_map.get(s.strip(), 3) for s in examples["sentiment"]]  # default to Irrelevant if unknown
    # Optional: entity-aware input for better performance
    examples["text"] = [f"Entity: {e}. {t}" for e, t in zip(examples["entity"], examples["text"])]
    return examples

dataset = dataset.map(encode_labels, batched=True)
dataset = dataset.remove_columns(["tweet_id", "entity", "sentiment"])  # clean up

# 3) Tokenizer & model (with local_files_only fix)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    local_files_only=True
)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # Trainer expects "labels"

# 4) Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5) Metric (macro F1 for multi-class)
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

# 6) Training args
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,  # GPU accel
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
