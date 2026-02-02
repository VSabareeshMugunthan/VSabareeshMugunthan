import torch
from transformers import pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# === Change these paths if needed ===
MODEL_PATH = r"C:\Users\sabar\OneDrive\Desktop\archive\cased-bert-dapt"
VALIDATION_CSV = r"C:\Users\sabar\OneDrive\Desktop\archive\twitter_validation.csv"

# Label mapping (must match what you used during training)
label_map = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
id2label = {v: k for k, v in label_map.items()}

print("Loading validation data...")
val = load_dataset(
    "csv",
    data_files={"val": VALIDATION_CSV},
    split="val",
    # Force column names because file probably has NO header
    column_names=["tweet_id", "entity", "sentiment", "text"]
)

# Preprocess same way as during training
def prepare(example):
    entity = example.get("entity", "Unknown")
    text = example.get("text", "")
    sentiment = example.get("sentiment", "Irrelevant").strip()
    
    example["text"] = f"Entity: {entity}. {text}"
    example["true_label"] = label_map.get(sentiment, 3)
    return example

val = val.map(prepare)

print(f"Loaded {len(val)} validation examples")

# Load model (use GPU if available)
print("Loading model...")
pipe = pipeline(
    "text-classification",
    model=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=32
)
# Replace the prediction loop with this version

print("Running predictions...")
preds = []

batch_size = 32
texts = list(val["text"])  # important conversion

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    
    # Predict batch
    batch_results = pipe(batch_texts)
    
    for res in batch_results:
        label_str = res["label"]          # 'LABEL_0', 'LABEL_1', ...
        pred_id = int(label_str.split("_")[1])
        preds.append(pred_id)
    
    # optional: show progress
    print(f"Processed {i + len(batch_texts)} / {len(texts)}", end="\r")

print(f"\nDone — {len(preds)} predictions completed")

# Compute metrics
print("Calculating metrics...")

# Make sure we have both lists ready
true_labels = val["true_label"]          # this should be a list of integers 0–3

# If you named it differently earlier, change to match, e.g.:
# true_labels = val["label"]   or whatever name you used

acc = evaluate.load("accuracy").compute(predictions=preds, references=true_labels)
prec = evaluate.load("precision").compute(predictions=preds, references=true_labels, average="macro")
rec = evaluate.load("recall").compute(predictions=preds, references=true_labels, average="macro")
f1_macro = evaluate.load("f1").compute(predictions=preds, references=true_labels, average="macro")
f1_weighted = evaluate.load("f1").compute(predictions=preds, references=true_labels, average="weighted")

print("\n" + "="*60)
print("FINAL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:          {acc['accuracy']:.4f}")
print(f"Precision (macro): {prec['precision']:.4f}")
print(f"Recall (macro):    {rec['recall']:.4f}")
print(f"F1 (macro):        {f1_macro['f1']:.4f}")
print(f"F1 (weighted):     {f1_weighted['f1']:.4f}")
print("="*60)
