"""
train_setfit_downsampled_minilm.py
---------------------------------
Binary classification of fall risk from discharge notes
using SetFit with MiniLM on a balanced dataset (1000 samples).
Includes extended evaluation (ROC, PR, confusion matrix heatmap, learning curves).
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
    average_precision_score
)
import datetime

from setfit import SetFitModel, SetFitTrainer, TrainingArguments

DATA_PATH   = r"D:\fall_risk_project\discharge_fall_dataset.parquet"
RANDOM_SEED = 42
N_SAMPLES   = 1000      # per class for balancing (500 pos + 500 neg = 1000 total)
TEST_SIZE   = 0.2      # 80/20 split

df = pd.read_parquet(DATA_PATH)
df["fall_label"] = df["fall_label"].fillna(0).astype(int)

# balance dataset (500 pos + 500 neg)
pos = df[df["fall_label"] == 1].sample(N_SAMPLES, random_state=RANDOM_SEED)
neg = df[df["fall_label"] == 0].sample(N_SAMPLES, random_state=RANDOM_SEED)
df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED)

# split into train/test
train_df, test_df = train_test_split(
    df_balanced,
    test_size=TEST_SIZE,
    stratify=df_balanced["fall_label"],
    random_state=RANDOM_SEED
)

train_ds = Dataset.from_pandas(train_df[["text", "fall_label"]].rename(columns={"fall_label": "label"}))
test_ds  = Dataset.from_pandas(test_df[["text", "fall_label"]].rename(columns={"fall_label": "label"}))

print("Data prepared")
print("Train size:", train_ds.num_rows, " Test size:", test_ds.num_rows)

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    model.model_body.to("cuda")
else:
    print("Running on CPU")

args = TrainingArguments(
    batch_size=16,
    num_epochs=5,
    seed=RANDOM_SEED
)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    metric="accuracy",
    batch_size=16,
    num_epochs=5,
    seed=RANDOM_SEED
)

print("Starting training...")
trainer.train()
print("Training finished")

metrics = trainer.evaluate()
print("Eval:", metrics)

y_true = test_df["fall_label"].tolist()
y_pred = trainer.model.predict(test_df["text"].tolist())
y_scores = trainer.model.predict_proba(test_df["text"].tolist())[:, 1]

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))

import seaborn as sns
import datetime
import os

# Get predicted probabilities
y_scores = trainer.model.predict_proba(test_df["text"].tolist())[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_BASE = r"D:\fall_risk_project\models"
MODEL_SAVE_PATH = os.path.join(MODEL_BASE, f"fall_risk_setfit_minilm_{timestamp}")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Save SetFit model
trainer.model.save_pretrained(MODEL_SAVE_PATH)

# Save metrics and reports
with open(os.path.join(MODEL_SAVE_PATH, "eval_metrics.txt"), "w") as f:
    f.write(str(metrics))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred, digits=3))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

# Save ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fall Risk Prediction")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(MODEL_SAVE_PATH, "roc_curve.png"))
plt.close()

# Save Precision-Recall Curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="purple", lw=2,
         label=f"PR (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(MODEL_SAVE_PATH, "precision_recall_curve.png"))
plt.close()

# Save Confusion Matrix Heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fall", "Fall"],
            yticklabels=["No Fall", "Fall"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(MODEL_SAVE_PATH, "confusion_matrix.png"))
plt.close()

print("Model, metrics, and plots saved to", MODEL_SAVE_PATH)
