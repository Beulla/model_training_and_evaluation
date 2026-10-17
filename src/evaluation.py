import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title(title)
    if fig:
        plt.tight_layout()
    return ax


def plot_roc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    title: str = "ROC Curves (One-vs-Rest)",
) -> plt.Figure:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        RocCurveDisplay.from_predictions(y_bin[:, i], y_proba[:, i], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def compute_multiclass_auc(
    y_true: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> float:
    y_bin = label_binarize(y_true, classes=range(n_classes))
    return roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str = "Model",
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )


def plot_learning_curve_keras(history, title: str = "Deep Learning Learning Curves"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "accuracy" in history.history:
        axes[1].plot(history.history["accuracy"], label="Train Acc")
        axes[1].plot(history.history["val_accuracy"], label="Val Acc")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Top Feature Importances",
) -> plt.Figure:
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        hue=[feature_names[i] for i in idx],
        ax=ax,
        legend=False,
        palette="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig
