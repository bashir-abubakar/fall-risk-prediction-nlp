"""
train_hf_trainer_balanced_10k.py
Binary classification of fall risk from discharge notes
using HuggingFace Trainer on a balanced dataset (5000 pos + 5000 neg).
Saves trained model and evaluation charts.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_PATH   = r"D:\fall_risk_project\discharge_fall_dataset.parquet"
MODEL_OUT   = r"D:\fall_risk_project\hf_trainer_fall_risk_balanced_10k_distillbert"
RANDOM_SEED = 42
N_SAMPLES   = 5000     # per class for balancing
TEST_SIZE   = 0.2      # 80/20 split

os.makedirs(MODEL_OUT, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["fall_label"] = df["fall_label"].fillna(0).astype(int)

# Balance dataset: 5000 pos + 5000 neg
pos = df[df["fall_label"] == 1].sample(N_SAMPLES, random_state=RANDOM_SEED)
neg = df[df["fall_label"] == 0].sample(N_SAMPLES, random_state=RANDOM_SEED)
df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED)

# Split
train_df, test_df = train_test_split(
    df_balanced,
    test_size=TEST_SIZE,
    stratify=df_balanced["fall_label"],
    random_state=RANDOM_SEED
)

# Convert to HuggingFace datasets
train_ds = Dataset.from_pandas(train_df[["text", "fall_label"]].rename(columns={"fall_label": "labels"}))
test_ds  = Dataset.from_pandas(test_df[["text", "fall_label"]].rename(columns={"fall_label": "labels"}))

print("Data prepared")
print("Train size:", train_ds.num_rows, " Test size:", test_ds.num_rows)

# 2. Tokenizer & model
MODEL_NAME = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    model.to("cuda")
else:
    print("Running on CPU")

args = TrainingArguments(
    output_dir=MODEL_OUT,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_OUT, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
)

from evaluate import load
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training finished")

trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
print("Model saved to", MODEL_OUT)

preds = trainer.predict(test_ds)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(axis=-1)
y_scores = preds.predictions[:, 1]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

with open(os.path.join(MODEL_OUT, "classification_report.txt"), "w") as f:
    f.write(classification_report(y_true, y_pred, digits=3))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred)))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(MODEL_OUT, "confusion_matrix.png"), dpi=300)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(MODEL_OUT, "roc_curve.png"), dpi=300)
plt.close()

precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="purple", lw=2, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(MODEL_OUT, "pr_curve.png"), dpi=300)
plt.close()

print("All metrics and plots saved in", MODEL_OUT)
