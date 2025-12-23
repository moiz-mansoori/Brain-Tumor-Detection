"""
VGG-16 Model Module

Transfer learning model using VGG-16 pretrained on ImageNet.
Custom classification head for multi-class tumor detection.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


def build_vgg16_model(
    input_shape: Tuple[int, int, int] = Config.INPUT_SHAPE,
    dense_units: int = Config.DENSE_UNITS,
    dropout_rate: float = Config.DROPOUT_RATE,
    freeze_base: bool = True
) -> Model:
    """
    Build VGG-16 model with custom classification head.
    
    Architecture:
    - VGG-16 base (ImageNet weights, no top)
    - GlobalAveragePooling2D
    - Dense(256) + ReLU
    - Dropout(0.5)
    - Dense(NUM_CLASSES) + Softmax
    
    Args:
        input_shape: Input image shape (height, width, channels)
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        freeze_base: If True, freeze VGG-16 base layers
        
    Returns:
        Compiled Keras model
    """
    logger.info("Building VGG-16 model...")
    
    # Load VGG-16 base model (pretrained on ImageNet)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base layers (Phase 1: transfer learning)
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
        logger.info("Base model layers frozen for transfer learning")
    
    # Build custom classification head
    x = base_model.output
    
    # Global Average Pooling (reduces spatial dimensions)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Batch Normalization for stability
    x = layers.BatchNormalization(name='batch_norm')(x)
    
    # Dense layer with ReLU activation
    x = layers.Dense(
        dense_units, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name='dense_hidden'
    )(x)
    
    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output layer (multi-class classification)
    output = layers.Dense(
        Config.NUM_CLASSES,  # 4 classes
        activation='softmax',
        name='output'
    )(x)
    
    # Create final model
    model = Model(
        inputs=base_model.input,
        outputs=output,
        name='VGG16_BrainTumor_4Class'
    )
    
    # Log model summary
    trainable_params = sum([tf.keras.backend.count_params(w) 
                          for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in model.non_trainable_weights])
    
    logger.info(f"Model built successfully!")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info(f"  Non-trainable params: {non_trainable_params:,}")
    
    return model


def compile_model(
    model: Model,
    learning_rate: float = Config.LEARNING_RATE_PHASE1
) -> Model:
    """
    Compile model with optimizer and loss function.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',  # Multi-class loss
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Model compiled with learning rate: {learning_rate}")
    return model


def unfreeze_model(
    model: Model,
    unfreeze_layers: int = Config.UNFREEZE_LAYERS,
    learning_rate: float = Config.LEARNING_RATE_PHASE2
) -> Model:
    """
    Unfreeze top layers of VGG-16 for fine-tuning (Phase 2).
    
    Args:
        model: Compiled Keras model
        unfreeze_layers: Number of layers to unfreeze from the top
        learning_rate: Lower learning rate for fine-tuning
        
    Returns:
        Model with unfrozen layers, recompiled
    """
    logger.info(f"Unfreezing last {unfreeze_layers} layers for fine-tuning...")
    
    # Find the base model layers
    # VGG-16 has 19 layers, we unfreeze the last few
    for layer in model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    for layer in model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate)
    
    # Log updated trainable status
    trainable_params = sum([tf.keras.backend.count_params(w) 
                          for w in model.trainable_weights])
    logger.info(f"Trainable params after unfreezing: {trainable_params:,}")
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


def load_model(model_path: str = None) -> Model:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to model file (.h5)
        
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        model_path = str(Config.MODEL_PATH)
    
    logger.info(f"Loading model from: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    logger.info("Model loaded successfully!")
    return model


def save_model(model: Model, model_path: str = None) -> str:
    """
    Save model to file.
    
    Args:
        model: Trained Keras model
        model_path: Output path (.h5)
        
    Returns:
        Path where model was saved
    """
    if model_path is None:
        model_path = str(Config.MODEL_PATH)
    
    # Ensure directory exists
    Config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return model_path


if __name__ == "__main__":
    print("VGG-16 Model Module Test")
    print("=" * 40)
    
    # Build model
    model = build_vgg16_model()
    model = compile_model(model)
    
    print("\nModel Summary:")
    print(get_model_summary(model))
    
    print("\nModule loaded successfully!")
