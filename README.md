# 🧠 Brain Tumor Detection using ML, CNN & Computer Vision

This is a practical machine learning project that detects brain tumors from MRI scans. It combines automated brain region extraction using computer vision with a fine-tuned VGG-16 convolutional neural network to classify images as tumor or non-tumor. The project includes preprocessing, training, evaluation, explainability, and a Streamlit demo app.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Model Status](https://img.shields.io/badge/Model-Custom_Trained_%26_Fine--Tuned-brightgreen)

---

## 🎯 Project Overview

This project features a **custom-trained and fine-tuned** VGG-16 model, specifically optimized for brain tumor detection. Unlike a generic pre-trained model, this system was trained through a rigorous two-phase pipeline using a custom dataset of thousands of MRI images.

Key highlights:
- **Phase 1 (Transfer Learning)**: Leveraging ImageNet weights for feature extraction.
- **Phase 2 (Fine-Tuning)**: Unfreezing top layers and training with a low learning rate for high-precision medical classification.

- **Automated Brain Extraction**: CV pipeline isolates brain regions from MRI scans
- **Transfer Learning**: VGG-16 pretrained on ImageNet, fine-tuned for tumor detection
- **Two-Phase Training**: Frozen base → fine-tuned layers for optimal performance
- **Medical-Context Metrics**: Evaluation focused on clinical relevance (recall, false negatives)
- **Cloud Training**: Google Colab notebook for GPU-accelerated training
- **Web Deployment**: Streamlit app for inference

> ✅ **Status**: The model is fully **trained and fine-tuned** with **99% accuracy** on the test set. It is ready for deployment.

---

## 📊 Dataset

| Class | Count | Percentage |
|-------|-------|------------|
| Glioma | 1621 | 23.1% |
| Meningioma | 1645 | 23.4% |
| No Tumor | 2000 | 28.5% |
| Pituitary | 1757 | 25.0% |
| **Total** | **7023** | 100% |

**Source**: Brain MRI Images for Brain Tumor Detection

---

## 🏗️ Architecture

### Computer Vision Pipeline
```
MRI Image → Grayscale → Gaussian Blur → Otsu Threshold → Contour Detection → Crop → Resize (224×224)
```

### VGG-16 Model
```
VGG-16 Base (ImageNet) → GlobalAvgPool → BatchNorm → Dense(256, ReLU) → Dropout(0.5) → Dense(4, softmax)
```

### 🧠 Explainability (Grad-CAM)

This project features a robust **Grad-CAM** (Gradient-weighted Class Activation Mapping) implementation in `src/utils/explainability.py`:

- ✅ **Heatmap Generation**: Visualizes which regions the CNN focuses on for predictions
- ✅ **Overlay Visualization**: Blends heatmap with original MRI for intuitive interpretation
- ✅ **Multi-class Support**: Works with all 4 tumor classes
- ✅ **Medical AI Trust**: Helps validate model decisions align with actual tumor locations

> Enable Grad-CAM in the Streamlit app via the sidebar checkbox to see model explainability in action.

---

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── app/
│   └── streamlit_app.py  # Web inference interface
├── data/
│   ├── splits/           # Train/val/test splits
│   └── processed/        # Brain-cropped images
├── notebooks/
│   └── Brain_Tumor_Detection_Colab.ipynb  # Google Colab training
├── saved_models/         # Trained .h5 models
├── src/
│   ├── preprocessing/    # CV pipeline & data loading
│   ├── models/           # CNN architecture
│   ├── training/         # Training pipeline
│   ├── evaluation/       # Metrics & visualization
│   └── utils/            # Config, Grad-CAM, & logging
└── requirements.txt      # Consolidated dependencies
```

---

## 🚀 Quick Start

### Option 1: Train on Google Colab (Recommended)

1. **Open Notebook**: Upload `notebooks/Brain_Tumor_Detection_Colab.ipynb` to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → T4 GPU
3. **Upload Dataset**: Follow notebook instructions to upload to Google Drive
4. **Run All Cells**: Execute the complete training pipeline
5. **Download Model**: Save the trained `.h5` file

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/moiz-mansoori/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app (requires trained model)
streamlit run app/streamlit_app.py
```

---

## ☁️ Training on Google Colab

The Colab notebook includes:

| Section | Description |
|---------|-------------|
| 0. Setup | GPU check, Drive mount |
| 1. Config | Hyperparameters, paths |
| 2. Dataset Analysis | Class distribution, samples |
| 3. Data Splitting | 70/15/15 stratified split |
| 4. Brain Extraction | CV pipeline demonstration |
| 5. Data Generators | Augmentation, VGG preprocessing |
| 6. Model Building | VGG-16 with custom head |
| 7. Phase 1 Training | Frozen base (transfer learning) |
| 8. Phase 2 Training | Fine-tuning top layers |
| 9. Visualization | Training curves |
| 10. Evaluation | Confusion matrix, ROC, metrics |
| 11. Download | Export model to local |

---

## 📈 Training Strategy

### Phase 1: Transfer Learning
- VGG-16 convolutional layers **frozen**
- Only custom head is trained
- Learning rate: `1e-3`
- Epochs: 10

### Phase 2: Fine-Tuning
- Last 4 layers **unfrozen**
- Lower learning rate: `1e-5`
- Epochs: 20
- Class weighting for imbalance

### Callbacks
- `EarlyStopping`: Patience 5
- `ReduceLROnPlateau`: Factor 0.5
- `ModelCheckpoint`: Save best model

---

## 📊 Evaluation Metrics

| Metric | Medical Context |
|--------|-----------------|
| **Accuracy** | Overall correctness |
| **Precision** | False alarm rate (healthy misdiagnosed) |
| **Recall** | Tumor detection rate (CRITICAL - missed tumors) |
| **F1-Score** | Balance between precision and recall |
| **Specificity** | Correctly identifying no-tumor cases |
| **AUC** | Overall discriminative ability |

> 🔴 **Recall is critical**: Missing a tumor (false negative) is dangerous in medical contexts.

---

## 🖥️ Streamlit App

The web interface provides:

1. **Upload MRI Image**
2. **View Original Image**
3. **View Cropped Brain Region**
4. **Prediction with Confidence Score**
5. **Medical Disclaimer**

### Running the App

```bash
cd app
streamlit run streamlit_app.py
```

---

## ⚙️ Configuration

All settings in `src/utils/config.py`:

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

## 📚 References

- VGG-16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- Transfer Learning: [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- Dataset: Brain MRI Images for Brain Tumor Detection (Kaggle)

---

