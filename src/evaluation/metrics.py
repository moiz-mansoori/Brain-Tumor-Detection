"""
Evaluation Metrics Module

Comprehensive evaluation metrics for medical image classification.
Updated for Multi-Class Classification (4 classes).
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import Model

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


# Medical context descriptions for metrics
METRIC_DESCRIPTIONS = {
    "accuracy": """
    **Accuracy**: Overall correctness of predictions across all 4 classes.
    - Medical Context: Percentage of correctly identified cases.
    """,
    
    "precision": """
    **Precision (Weighted)**: Average precision across all classes.
    - Medical Context: When the model predicts a specific tumor type, how often is it right?
    - High precision reduces false alarms for specific conditions.
    """,
    
    "recall": """
    **Recall (Sensitivity/Weighted)**: Average recall across all classes.
    - Medical Context: What percentage of actual conditions did we catch?
    - High recall means we rarely miss a diagnosis.
    """,
    
    "f1_score": """
    **F1-Score**: Harmonic mean of precision and recall.
    - Medical Context: Balanced measure for overall performance.
    """,
    
    "auc": """
    **AUC (One-vs-Rest)**: Average Area Under the ROC Curve.
    - Medical Context: Ability of the model to distinguish between classes.
    """
}


def evaluate_model(
    model: Model,
    test_generator,
    threshold: float = 0.5  # Not used for multi-class argmax
) -> Dict[str, float]:
    """
    Comprehensive model evaluation on test data [Multi-Class].
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        threshold: Unused in multi-class (argmax used instead)
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Get predictions
    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=1)
    
    # Convert probabilities to class indices (Multi-class)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Compute metrics (Weighted average for multi-class)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC-AUC (One-vs-Rest for Multi-class)
    try:
        # Binarize labels for AUC
        y_true_bin = label_binarize(y_true, classes=range(len(Config.CLASSES)))
        metrics["auc"] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        metrics["auc"] = 0.0
    
    # Log results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS (Multi-Class)")
    logger.info("=" * 50)
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1-Score:    {metrics['f1_score']:.4f}")
    logger.info(f"AUC (OvR):   {metrics['auc']:.4f}")
    logger.info("=" * 50)
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=Config.CLASSES)
    logger.info("\nClassification Report:\n" + report)
    
    return metrics


def plot_confusion_matrix(
    model: Model,
    test_generator,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix for multi-class.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        threshold: Unused
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get predictions
    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=Config.CLASSES,
        yticklabels=Config.CLASSES,
        ax=ax,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Brain Tumor Detection', fontsize=14)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(
    model: Model,
    test_generator,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curve for multi-class (One-vs-Rest).
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Get predictions
    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=0)
    y_true = test_generator.classes
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(Config.CLASSES)))
    n_classes = len(Config.CLASSES)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC for each class
    colors = ['blue', 'orange', 'green', 'red']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{Config.CLASSES[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, 
            label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")
    
    return fig


def generate_evaluation_report(
    metrics: dict,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Optional path to save report
        
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("BRAIN TUMOR DETECTION - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Performance summary
    report.append("## Performance Summary (Weighted Average)")
    report.append("-" * 40)
    report.append(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    report.append(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    report.append(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    report.append(f"F1-Score:    {metrics['f1_score']:.4f}")
    if 'auc' in metrics:
        report.append(f"AUC (OvR):   {metrics['auc']:.4f}")
    report.append("")
    
    # Disclaimer
    report.append("## Disclaimer")
    report.append("-" * 40)
    report.append("This model is for educational/demonstration purposes only.")
    report.append("It is NOT intended for clinical diagnosis or medical decisions.")
    
    report_text = "\n".join(report)
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Evaluation report saved to: {save_path}")
    
    return report_text


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1 = axes[0]
    ax1.plot(history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(history['val_accuracy'], label='Val Accuracy', color='orange')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2 = axes[1]
    ax2.plot(history['loss'], label='Train Loss', color='blue')
    ax2.plot(history['val_loss'], label='Val Loss', color='orange')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Brain Tumor Detection', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {save_path}")
    
    return fig


def print_metric_explanations():
    """Print medical context explanations for all metrics."""
    print("\n" + "=" * 60)
    print("METRIC EXPLANATIONS (Medical Context)")
    print("=" * 60)
    
    for metric, description in METRIC_DESCRIPTIONS.items():
        print(f"\n{description}")


if __name__ == "__main__":
    print("Metrics Module Test")
    print("=" * 40)
    print_metric_explanations()
    print("\nModule loaded successfully!")
