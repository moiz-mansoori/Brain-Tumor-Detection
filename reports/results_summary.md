# Brain Tumor Detection - Results Summary

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | VGG-16 (ImageNet pretrained) |
| Input Size | 224×224×3 |
| Batch Size | 32 |
| Phase 1 Epochs | 10 (frozen base) |
| Phase 2 Epochs | 20 (fine-tuning) |
| Learning Rate P1 | 1e-3 |
| Learning Rate P2 | 1e-5 |
| Dropout | 0.5 |
| Dense Units | 256 |

## Dataset Split

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|-------|--------|------------|----------|-----------|-------|
| Train | 1378   | 1399       | 1700     | 1494      | 5971  |
| Val   | 243    | 246        | 300      | 263       | 1052  |
| Test  | 300    | 306        | 405      | 300       | 1311  |

## 4. Evaluation Metrics (Test Set)

| Metric | Value | Notes |
|:-------|:------|:------|
| **Accuracy** | **99%** | Overall correct predictions |
| **Precision (Macro)** | **0.99** | High precision across all classes |
| **Recall (Macro)** | **0.99** | Excellent sensitivity (few missed tumors) |
| **F1-Score (Macro)** | **0.99** | Balanced performance |
| **Validation Loss** | **0.0483** | Low loss indicates good convergence |
| **Training Accuracy** | **99.39%** | Very close to validation (98.55%), minimal overfitting |

## 5. Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------|:-------|:---------|:--------|
| **Glioma** | 0.98 | 0.98 | 0.98 | 300 |
| **Meningioma** | 0.98 | 0.97 | 0.98 | 306 |
| **No Tumor** | **1.00** | **1.00** | **1.00** | 405 |
| **Pituitary** | 0.98 | 0.99 | 0.99 | 300 |

> **Analysis**: The model performs exceptionally well. Ideally, "No Tumor" detection is 100% accurate, meaning healthy patients are reliably identified. Tumor classes also have very high recall (>97%), minimizing dangerous false negatives.

## 6. Confusion Matrix

*(Refer to the generated plot in `notebooks/Brain_Tumor_4Class_Colab.ipynb` for the visual matrix)*

Key observations:
- **No Tumor** class is perfectly separated.
- Minimal confusion between tumor types (Glioma/Meningioma/Pituitary).
```
                  Predicted
              No Tumor | Tumor
Actual   No Tumor  [TN]  | [FP]
         Tumor     [FN]  | [TP]
```

- **True Positives (TP)**: Tumors correctly detected
- **True Negatives (TN)**: No-tumor correctly identified  
- **False Positives (FP)**: False alarms

