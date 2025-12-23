"""
Data Loader Module

TensorFlow/Keras data generators with augmentation support.
Properly configured for VGG-16 preprocessing.
"""

from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
# TensorFlow 2.16+ compatibility
try:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.vgg16 import preprocess_input
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import preprocess_input

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


def create_generators(
    data_dir: Path = None,
    batch_size: int = Config.BATCH_SIZE,
    target_size: Tuple[int, int] = Config.IMAGE_SIZE,
    augment_train: bool = True,
    use_colab: bool = False
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, 
           tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator]:
    """
    Create train, validation, and test data generators.
    
    Args:
        data_dir: Base data directory (with train/val/test subdirs)
        batch_size: Batch size for training
        target_size: Image target size (width, height)
        augment_train: Whether to apply augmentation to training data
        use_colab: If True, use Colab paths
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    if data_dir is None:
        paths = Config.get_paths(use_colab)
        data_dir = paths["processed_dir"]
    
    data_dir = Path(data_dir)
    
    # Training generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=Config.AUGMENTATION_PARAMS["rotation_range"],
            width_shift_range=Config.AUGMENTATION_PARAMS["width_shift_range"],
            height_shift_range=Config.AUGMENTATION_PARAMS["height_shift_range"],
            shear_range=Config.AUGMENTATION_PARAMS["shear_range"],
            zoom_range=Config.AUGMENTATION_PARAMS["zoom_range"],
            horizontal_flip=Config.AUGMENTATION_PARAMS["horizontal_flip"],
            vertical_flip=Config.AUGMENTATION_PARAMS["vertical_flip"],
            fill_mode=Config.AUGMENTATION_PARAMS["fill_mode"]
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    
    # Validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir / "train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # Multi-class
        classes=Config.CLASSES,
        shuffle=True,
        seed=Config.RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        data_dir / "val",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # Multi-class
        classes=Config.CLASSES,
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        data_dir / "test",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # Multi-class
        classes=Config.CLASSES,
        shuffle=False
    )
    
    # Log generator info
    logger.info(f"Training samples: {train_generator.samples}")
    logger.info(f"Validation samples: {val_generator.samples}")
    logger.info(f"Test samples: {test_generator.samples}")
    logger.info(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


def create_single_generator(
    data_dir: Path,
    batch_size: int = Config.BATCH_SIZE,
    target_size: Tuple[int, int] = Config.IMAGE_SIZE,
    augment: bool = False,
    shuffle: bool = True
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Create a single data generator for a directory.
    
    Args:
        data_dir: Directory with yes/no subdirectories
        batch_size: Batch size
        target_size: Image target size
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle data
        
    Returns:
        Data generator
    """
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **Config.AUGMENTATION_PARAMS
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # Multi-class
        classes=Config.CLASSES,
        shuffle=shuffle,
        seed=Config.RANDOM_SEED
    )
    
    return generator


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = Config.IMAGE_SIZE
) -> tf.Tensor:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=target_size
    )
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)
    
    # Apply VGG-16 preprocessing
    img_array = preprocess_input(img_array)
    
    return img_array


def get_class_weights(train_generator) -> dict:
    """
    Compute class weights from training generator.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        Dictionary mapping class index to weight
    """
    from collections import Counter
    import numpy as np
    
    # Get all labels
    labels = train_generator.classes
    
    # Count classes
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    
    # Compute weights
    weights = {}
    for class_idx, count in counter.items():
        weights[class_idx] = total / (n_classes * count)
    
    logger.info(f"Class weights computed: {weights}")
    return weights


if __name__ == "__main__":
    print("Data Loader Module Test")
    print("=" * 40)
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Image size: {Config.IMAGE_SIZE}")
    print(f"Augmentation params: {Config.AUGMENTATION_PARAMS}")
    print("Module loaded successfully!")
