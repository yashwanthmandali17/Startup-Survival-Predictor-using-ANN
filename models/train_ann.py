"""
train_ann.py
Trains an ANN (MLPClassifier) on the startup dataset.
Saves the model, scaler, and encoders as .pkl files.
"""
import pandas as pd
import numpy as np
import joblib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

import sys
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.eda_preprocessing import load_and_eda, preprocess

MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "data", "eda_plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


def train_and_evaluate(csv_path: str):
    # ─── Load & preprocess ────────────────────────────────────────────
    df = load_and_eda(csv_path, save_plots=True, plot_dir=PLOT_DIR)
    X, y, scaler, le_dict, target_le, feature_names = preprocess(df)

    # ─── Train/test split (stratified) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─── Class weights to handle imbalance ───────────────────────────
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, cw))

    # ─── ANN Architecture ────────────────────────────────────────────
    ann = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )

    print("\n⚙  Training ANN  (128 → 64 → 32 → softmax) ...")
    ann.fit(X_train, y_train)
    print(f"   Converged after {ann.n_iter_} iterations")

    # ─── Evaluation ──────────────────────────────────────────────────
    y_pred = ann.predict(X_test)
    y_prob = ann.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY : {acc*100:.2f}%")
    print(f"  ROC-AUC (OVR) : {roc_auc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=target_le.classes_))

    # ─── Cross-validation ────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ann, X, y, cv=cv, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ─── Plots ───────────────────────────────────────────────────────
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_le.classes_,
                yticklabels=target_le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix  (Acc={acc*100:.1f}%)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/07_confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ann.loss_curve_, label="Train Loss", color="#5C6BC0", linewidth=2)
    if hasattr(ann, "validation_scores_") and ann.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(ann.validation_scores_, label="Val Accuracy",
                 color="#FF7043", linewidth=2, linestyle="--")
        ax2.set_ylabel("Validation Accuracy", color="#FF7043")
        ax2.legend(loc="lower right")
    ax.set_title("ANN Training Loss Curve")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Loss", color="#5C6BC0")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_loss_curve.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Feature importance via permutation proxy
    importances = np.abs(ann.coefs_[0]).mean(axis=1)
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    feat_imp.plot(kind="barh", ax=ax, color="#5C6BC0")
    ax.set_title("Feature Importance (Input-layer weight magnitude)")
    ax.set_xlabel("Mean Absolute Weight")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/09_feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ─── Save artifacts ──────────────────────────────────────────────
    artifacts = {
        "model": ann,
        "scaler": scaler,
        "label_encoders": le_dict,
        "target_le": target_le,
        "feature_names": feature_names,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "class_names": list(target_le.classes_)
    }
    # Strip internal numpy random state to avoid BitGenerator version conflicts
    import copy
    safe_model = copy.deepcopy(ann)
    if hasattr(safe_model, '_no_improvement_count'):
        pass  # fine
    # Reset the random state to avoid numpy version conflicts on load
    from sklearn.utils import check_random_state
    safe_model._random_state = None
    artifacts["model"] = safe_model
    joblib.dump(artifacts, os.path.join(MODEL_DIR, "ann_model.pkl"))

    print(f"\n✓ Model saved to {os.path.join(MODEL_DIR, 'ann_model.pkl')}")
    print(f"✓ Plots saved to {PLOT_DIR}")
    return artifacts


if __name__ == "__main__":
    train_and_evaluate("D:/try-2/data/startup_data.csv")
