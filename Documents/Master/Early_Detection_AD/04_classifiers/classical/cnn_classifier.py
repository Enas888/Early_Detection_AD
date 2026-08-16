"""
cnn_classifier.py

Multiclass Alzheimer's classification:

CN
EMCI
LMCI
AD

Uses:
    - Frozen ResNet18 backbone
    - Train only embedding head + classifier
    - Class-weighted CrossEntropy
    - Best model selected by Validation Macro F1

Realistic evaluation additions:
    - Per-class F1 / Precision / Recall (not just macro average)
    - Weighted F1 + Balanced Accuracy (cross-checks against Macro F1)
    - Confusion matrix (raw + normalized) on validation AND held-out test
    - Bootstrap 95% confidence interval on test Macro F1
      (cheaper alternative to full k-fold, still gives uncertainty estimate)
    - Final classification report (precision/recall/f1 per class) saved to disk

Outputs:

03_feature_extraction/deep/models/
    best_resnet18_8d_multiclass.pth

08_results/tables/
    cnn_training_history.csv
    cnn_training_summary.csv
    cnn_model_info.csv
    cnn_val_classification_report.csv
    cnn_test_classification_report.csv
    cnn_test_bootstrap_ci.csv

08_results/figures/
    cnn_loss_curve.png
    cnn_accuracy_curve.png
    cnn_f1_curve.png
    cnn_val_confusion_matrix.png
    cnn_test_confusion_matrix.png
"""

from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(
    "/home/enooo/Documents/Master/Early_Detection_AD"
)

sys.path.append(
    str(
        PROJECT_ROOT
        / "03_feature_extraction"
        / "deep"
    )
)

from dataset import SliceDataset
from cnn_extractor import (
    AlzheimerResNet18,
    count_parameters
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-3
N_BOOTSTRAP = 1000          # bootstrap resamples for test-set CI
RANDOM_SEED = 42

CLASS_NAMES = ["CN", "EMCI", "LMCI", "AD"]   # adjust order to match your label encoding

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device:", DEVICE)
print("=" * 60)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

METADATA_CSV = (
    PROJECT_ROOT / "01_data" / "csv" / "slice_metadata.csv"
)

MODEL_DIR = (
    PROJECT_ROOT / "03_feature_extraction" / "deep" / "models"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_TABLES = PROJECT_ROOT / "08_results" / "tables"
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

RESULTS_FIGURES = PROJECT_ROOT / "08_results" / "figures"
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = SliceDataset(METADATA_CSV, split="train")
val_dataset   = SliceDataset(METADATA_CSV, split="val")
test_dataset  = SliceDataset(METADATA_CSV, split="test")   # held out, used ONLY at the end

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

class_counts = (
    train_dataset.df["label"]
    .value_counts()
    .sort_index()
)

num_classes = len(class_counts)
total_samples = class_counts.sum()

weights = total_samples / (num_classes * class_counts)
weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

print("\nClass weights:")
print(weights)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = AlzheimerResNet18()

for param in model.backbone.parameters():
    param.requires_grad = False

model = model.to(DEVICE)

total_params, trainable_params = count_parameters(model)

print("\nModel parameters")
print("Total:", f"{total_params:,}")
print("Trainable:", f"{trainable_params:,}")

# --------------------------------------------------
# LOSS / OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# --------------------------------------------------
# HELPER: RUN ONE EPOCH (train or eval)
# --------------------------------------------------

def run_epoch(loader, train_mode):

    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:
        for batch in loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)

    return avg_loss, np.array(all_targets), np.array(all_preds)


def compute_metrics(targets, preds):
    """Returns a dict of realistic, multi-angle metrics — not just Macro F1."""

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    kappa = cohen_kappa_score(targets, preds)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
    }


def plot_confusion_matrix(targets, preds, class_names, title, save_path):
    """Saves both raw-count and row-normalized (recall-style) confusion matrices."""

    cm = confusion_matrix(targets, preds)
    cm_norm = confusion_matrix(targets, preds, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_title(f"{title} — Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_title(f"{title} — Normalized (Recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def bootstrap_macro_f1_ci(targets, preds, n_bootstrap=1000, alpha=0.05, seed=42):
    """
    Cheap alternative to full k-fold cross-validation: resample the
    held-out test predictions with replacement many times and recompute
    Macro F1 each time, to get an uncertainty estimate around the
    reported test performance.
    """
    rng = np.random.RandomState(seed)
    n = len(targets)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_targets = targets[idx]
        boot_preds = preds[idx]
        scores.append(f1_score(boot_targets, boot_preds, average="macro"))

    scores = np.array(scores)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = scores.mean()

    return mean_score, lower, upper, scores


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------

history = []
best_macro_f1 = 0.0
best_epoch = 0

start_time = time.time()

for epoch in range(EPOCHS):

    train_loss, train_targets, train_preds = run_epoch(train_loader, train_mode=True)
    train_metrics = compute_metrics(train_targets, train_preds)

    val_loss, val_targets, val_preds = run_epoch(val_loader, train_mode=False)
    val_metrics = compute_metrics(val_targets, val_preds)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss={train_loss:.4f} Acc={train_metrics['accuracy']:.4f} | "
        f"Val Loss={val_loss:.4f} MacroF1={val_metrics['macro_f1']:.4f} "
        f"BalAcc={val_metrics['balanced_accuracy']:.4f} "
        f"WeightedF1={val_metrics['weighted_f1']:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_metrics["accuracy"],
        "val_acc": val_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_balanced_acc": val_metrics["balanced_accuracy"],
        "val_kappa": val_metrics["cohen_kappa"],
    })

    # ==========================================
    # SAVE BEST MODEL (selection metric: Macro F1)
    # ==========================================

    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = val_metrics["macro_f1"]
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_resnet18_8d_multiclass.pth"
        )

        # cache best-epoch validation predictions for the final report
        best_val_targets = val_targets.copy()
        best_val_preds = val_preds.copy()

        print("Saved best model.")

training_minutes = (time.time() - start_time) / 60.0

print("\nTraining completed.")
print(f"Best Macro F1 (val): {best_macro_f1:.4f}  (epoch {best_epoch})")

# --------------------------------------------------
# HISTORY CSV
# --------------------------------------------------

history_df = pd.DataFrame(history)
history_df.to_csv(RESULTS_TABLES / "cnn_training_history.csv", index=False)

# --------------------------------------------------
# VALIDATION — DETAILED REPORT AT BEST EPOCH
# --------------------------------------------------

val_report_dict = classification_report(
    best_val_targets, best_val_preds,
    target_names=CLASS_NAMES, output_dict=True, zero_division=0
)
val_report_df = pd.DataFrame(val_report_dict).transpose()
val_report_df.to_csv(RESULTS_TABLES / "cnn_val_classification_report.csv")

print("\n=== Validation classification report (best epoch) ===")
print(classification_report(best_val_targets, best_val_preds, target_names=CLASS_NAMES, zero_division=0))

plot_confusion_matrix(
    best_val_targets, best_val_preds, CLASS_NAMES,
    title="Validation Confusion Matrix",
    save_path=RESULTS_FIGURES / "cnn_val_confusion_matrix.png"
)

# --------------------------------------------------
# RELOAD BEST MODEL FOR FINAL, UNBIASED TEST EVALUATION
# --------------------------------------------------
# Important: never select hyperparameters/epoch based on test performance.
# The test set is touched exactly once, here, after training is fully done.

model.load_state_dict(torch.load(MODEL_DIR / "best_resnet18_8d_multiclass.pth"))
model = model.to(DEVICE)
model.eval()

test_loss, test_targets, test_preds = run_epoch(test_loader, train_mode=False)
test_metrics = compute_metrics(test_targets, test_preds)

print("\n=== FINAL HELD-OUT TEST SET RESULTS ===")
print(f"Test Loss          : {test_loss:.4f}")
print(f"Test Accuracy      : {test_metrics['accuracy']:.4f}")
print(f"Test Balanced Acc  : {test_metrics['balanced_accuracy']:.4f}")
print(f"Test Macro F1      : {test_metrics['macro_f1']:.4f}")
print(f"Test Weighted F1   : {test_metrics['weighted_f1']:.4f}")
print(f"Test Cohen's Kappa : {test_metrics['cohen_kappa']:.4f}")

test_report_dict = classification_report(
    test_targets, test_preds,
    target_names=CLASS_NAMES, output_dict=True, zero_division=0
)
test_report_df = pd.DataFrame(test_report_dict).transpose()
test_report_df.to_csv(RESULTS_TABLES / "cnn_test_classification_report.csv")

print("\n=== Test classification report (per-class) ===")
print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES, zero_division=0))

plot_confusion_matrix(
    test_targets, test_preds, CLASS_NAMES,
    title="Test Confusion Matrix",
    save_path=RESULTS_FIGURES / "cnn_test_confusion_matrix.png"
)

# --------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVAL ON TEST MACRO F1
# --------------------------------------------------
# Cheaper alternative to full k-fold CV — gives an uncertainty estimate
# around the reported test Macro F1 without retraining the model.

mean_f1, ci_lower, ci_upper, boot_scores = bootstrap_macro_f1_ci(
    test_targets, test_preds, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_SEED
)

print(f"\nBootstrap Macro F1 (test, n={N_BOOTSTRAP}): "
      f"{mean_f1:.4f}  95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")

bootstrap_df = pd.DataFrame([{
    "metric": "macro_f1",
    "point_estimate": test_metrics["macro_f1"],
    "bootstrap_mean": mean_f1,
    "ci_lower_95": ci_lower,
    "ci_upper_95": ci_upper,
    "n_bootstrap": N_BOOTSTRAP,
}])
bootstrap_df.to_csv(RESULTS_TABLES / "cnn_test_bootstrap_ci.csv", index=False)

plt.figure(figsize=(7, 4))
plt.hist(boot_scores, bins=40, color="#4C72B0", alpha=0.8)
plt.axvline(test_metrics["macro_f1"], color="red", linestyle="--", label="Point estimate")
plt.axvline(ci_lower, color="black", linestyle=":", label="95% CI")
plt.axvline(ci_upper, color="black", linestyle=":")
plt.xlabel("Bootstrap Macro F1")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Test Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_FIGURES / "cnn_test_bootstrap_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# SUMMARY CSV
# --------------------------------------------------

summary_df = pd.DataFrame([{
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_macro_f1,
    "test_accuracy": test_metrics["accuracy"],
    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    "test_macro_f1": test_metrics["macro_f1"],
    "test_weighted_f1": test_metrics["weighted_f1"],
    "test_cohen_kappa": test_metrics["cohen_kappa"],
    "test_macro_f1_ci_lower": ci_lower,
    "test_macro_f1_ci_upper": ci_upper,
    "training_minutes": training_minutes,
    "epochs": EPOCHS,
}])
summary_df.to_csv(RESULTS_TABLES / "cnn_training_summary.csv", index=False)

# --------------------------------------------------
# MODEL INFO CSV
# --------------------------------------------------

info_df = pd.DataFrame([{
    "architecture": "ResNet18",
    "latent_dim": 8,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
}])
info_df.to_csv(RESULTS_TABLES / "cnn_model_info.csv", index=False)

# --------------------------------------------------
# TRAINING CURVE PLOTS
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_FIGURES / "cnn_loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_acc"], label="Train")
plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_FIGURES / "cnn_accuracy_curve.png", dpi=150)
plt.close()

# New: F1 curve (Macro vs Weighted) — shows whether the model is doing
# well on the bulk of the data vs. the rare classes, across training
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_macro_f1"], label="Train Macro F1")
plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro F1")
plt.plot(history_df["epoch"], history_df["val_weighted_f1"], label="Val Weighted F1", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_FIGURES / "cnn_f1_curve.png", dpi=150)
plt.close()

print("\nSaved all results, reports, and figures.")
print(f"Tables  -> {RESULTS_TABLES}")
print(f"Figures -> {RESULTS_FIGURES}")