"""
Centralized Configuration Module

All hyperparameters, paths, and settings are defined here.
No hardcoded values elsewhere in the codebase.
"""

import os
from pathlib import Path


class Config:
    """Centralized configuration for the Brain Tumor Detection project."""
    
    # ==================== PATHS ====================
    # Base project directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    SPLITS_DIR = DATA_DIR / "splits"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Original dataset location (new multi-class dataset)
    ORIGINAL_DATASET_DIR = PROJECT_ROOT / "Dataset"
    TRAINING_DIR = ORIGINAL_DATASET_DIR / "Training"
    TESTING_DIR = ORIGINAL_DATASET_DIR / "Testing"
    
    # Model directory
    SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
    MODEL_PATH = SAVED_MODELS_DIR / "vgg16_brain_tumor_4class.h5"
    
    # Reports directory
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    # ==================== DATA SPLIT ====================
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ==================== IMAGE SETTINGS ====================
    IMAGE_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    
    # ==================== CLASS LABELS (4-class) ====================
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    CLASS_LABELS = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    NUM_CLASSES = 4
    
    # Class display names for UI
    CLASS_DISPLAY_NAMES = {
        "glioma": "Glioma Tumor",
        "meningioma": "Meningioma Tumor",
        "notumor": "No Tumor",
        "pituitary": "Pituitary Tumor"
    }
    
    # Colors for UI display
    CLASS_COLORS = {
        "glioma": "#E53935",      # Red
        "meningioma": "#FB8C00",  # Orange
        "notumor": "#43A047",     # Green
        "pituitary": "#1E88E5"    # Blue
    }
    
    # ==================== TRAINING HYPERPARAMETERS ====================
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 10  # Frozen base layers
    EPOCHS_PHASE2 = 20  # Fine-tuning
    LEARNING_RATE_PHASE1 = 1e-3
    LEARNING_RATE_PHASE2 = 1e-5
    
    # ==================== MODEL ARCHITECTURE ====================
    DENSE_UNITS = 256
    DROPOUT_RATE = 0.5
    
    # VGG-16 fine-tuning: unfreeze last N convolutional blocks
    UNFREEZE_LAYERS = 4  # Last 4 layers of VGG-16
    
    # ==================== CALLBACKS ====================
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # ==================== REPRODUCIBILITY ====================
    RANDOM_SEED = 42
    
    # ==================== DATA AUGMENTATION ====================
    AUGMENTATION_PARAMS = {
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
        "zoom_range": 0.1,
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "nearest"
    }
    
    # ==================== BRAIN CROPPING (CV) ====================
    BLUR_KERNEL_SIZE = (5, 5)
    THRESHOLD_METHOD = "otsu"  # Options: "otsu", "adaptive"
    MIN_CONTOUR_AREA = 1000  # Minimum area to consider as brain region
    PADDING = 10  # Pixels to add around bounding box
    
    # ==================== COLAB/CLOUD SETTINGS ====================
    # Google Drive mount path (when using Colab)
    COLAB_DRIVE_PATH = "/content/drive/MyDrive/Brain_Tumor_Detection"
    USE_COLAB = False  # Set to True when running in Google Colab
    
    @classmethod
    def get_paths(cls, use_colab: bool = False) -> dict:
        """
        Get appropriate paths based on environment.
        
        Args:
            use_colab: If True, return Colab/Drive paths
            
        Returns:
            Dictionary of paths
        """
        if use_colab:
            base = Path(cls.COLAB_DRIVE_PATH)
            return {
                "data_dir": base / "data",
                "raw_dir": base / "data" / "raw",
                "splits_dir": base / "data" / "splits",
                "processed_dir": base / "data" / "processed",
                "model_dir": base / "saved_models",
                "model_path": base / "saved_models" / "vgg16_brain_tumor_4class.h5"
            }
        else:
            return {
                "data_dir": cls.DATA_DIR,
                "raw_dir": cls.RAW_DATA_DIR,
                "splits_dir": cls.SPLITS_DIR,
                "processed_dir": cls.PROCESSED_DIR,
                "model_dir": cls.SAVED_MODELS_DIR,
                "model_path": cls.MODEL_PATH
            }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary project directories."""
        directories = [
            cls.DATA_DIR,
            cls.SAVED_MODELS_DIR,
            cls.REPORTS_DIR
        ]
        
        # Add directories for each class
        for split in ["train", "val", "test"]:
            for class_name in cls.CLASSES:
                directories.append(cls.SPLITS_DIR / split / class_name)
                directories.append(cls.PROCESSED_DIR / split / class_name)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def display(cls) -> None:
        """Display current configuration."""
        print("=" * 50)
        print("BRAIN TUMOR DETECTION - CONFIGURATION")
        print("=" * 50)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Classes: {cls.CLASSES} ({cls.NUM_CLASSES} classes)")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Random Seed: {cls.RANDOM_SEED}")
        print(f"Model Path: {cls.MODEL_PATH}")
        print("=" * 50)


if __name__ == "__main__":
    # Test configuration
    Config.display()
    Config.create_directories()
    print("\nDirectories created successfully!")
