"""
train_setfit_downsampled.py
---------------------------
Binary classification of fall risk from discharge notes
using SetFit on a **downsampled** balanced dataset (2000 train / 500 test).
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import datetime

from setfit import SetFitModel, SetFitTrainer, TrainingArguments

DATA_PATH   = r"D:\fall_risk_project\discharge_fall_dataset.parquet"
MODEL_OUT   = r"D:\fall_risk_project\fall_risk_setfit_model_downsampled"
RANDOM_SEED = 42
N_SAMPLES   = 5000     # per class for balancing
TEST_SIZE   = 0.2      # 80/20 split
TRAIN_KEEP  = 2000     # keep 2000 train examples
TEST_KEEP   = 500      # keep 500 test examples

df = pd.read_parquet(DATA_PATH)
df["fall_label"] = df["fall_label"].fillna(0).astype(int)

# balance dataset (5000 pos + 5000 neg)
pos = df[df["fall_label"] == 1].sample(N_SAMPLES, random_state=RANDOM_SEED)
neg = df[df["fall_label"] == 0].sample(N_SAMPLES, random_state=RANDOM_SEED)
df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED)

# split
train_df, test_df = train_test_split(
    df_balanced,
    test_size=TEST_SIZE,
    stratify=df_balanced["fall_label"],
    random_state=RANDOM_SEED
)

train_ds = Dataset.from_pandas(train_df[["text", "fall_label"]].rename(columns={"fall_label": "label"}))
test_ds  = Dataset.from_pandas(test_df[["text", "fall_label"]].rename(columns={"fall_label": "label"}))

print("Data prepared")
print("Full Train size:", train_ds.num_rows, " Full Test size:", test_ds.num_rows)

train_ds_small = train_ds.shuffle(seed=RANDOM_SEED).select(range(TRAIN_KEEP))
test_ds_small  = test_ds.shuffle(seed=RANDOM_SEED).select(range(TEST_KEEP))

print("Downsampled Train size:", train_ds_small.num_rows, " Downsampled Test size:", test_ds_small.num_rows)

# Stronger general-purpose model
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")


if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    model.model_body.to("cuda")
else:
    print("Running on CPU")

args = TrainingArguments(
    batch_size=16,
    num_epochs=3,
    seed=RANDOM_SEED
)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds_small,
    eval_dataset=test_ds_small,
    metric="accuracy",
    batch_size=16,
    num_epochs=3,   # or 5 if you want longer training
    seed=RANDOM_SEED
)


print("Starting training...")
trainer.train()
print("Training finished")


metrics = trainer.evaluate()
print("Eval:", metrics)

y_true = test_df.sample(TEST_KEEP, random_state=RANDOM_SEED)["fall_label"].tolist()
y_pred = trainer.model.predict(test_df.sample(TEST_KEEP, random_state=RANDOM_SEED)["text"].tolist())

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))


y_scores = trainer.model.predict_proba(test_df.sample(TEST_KEEP, random_state=RANDOM_SEED)["text"].tolist())[:, 1]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fall Risk Prediction")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Create unique output folder with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = os.path.join(
    r"D:\fall_risk_project", f"fall_risk_setfit_model_downsampled_{timestamp}"
)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Save SetFit model
trainer.model.save_pretrained(MODEL_SAVE_PATH)

# Save training args & metrics
with open(os.path.join(MODEL_SAVE_PATH, "eval_metrics.txt"), "w") as f:
    f.write(str(metrics))

print("Model + artifacts saved to", MODEL_SAVE_PATH)
