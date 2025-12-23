"""
Trainer Module

Complete training pipeline with callbacks, logging, and two-phase training.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.seed import set_seed
from ..models.vgg16_model import (
    build_vgg16_model,
    compile_model,
    unfreeze_model,
    save_model
)
from ..preprocessing.data_loader import get_class_weights


logger = get_logger(__name__)


class Trainer:
    """
    Training manager for the brain tumor detection model.
    
    Supports:
    - Two-phase training (frozen → fine-tuned)
    - Automatic class weight handling
    - Comprehensive callbacks
    - Training history logging
    """
    
    def __init__(
        self,
        model: Model = None,
        use_colab: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Pre-built model (if None, builds new model)
            use_colab: If True, use Colab paths
        """
        set_seed(Config.RANDOM_SEED)
        
        self.use_colab = use_colab
        self.paths = Config.get_paths(use_colab)
        
        # Build or use provided model
        if model is None:
            self.model = build_vgg16_model(freeze_base=True)
            self.model = compile_model(self.model)
        else:
            self.model = model
        
        self.history_phase1 = None
        self.history_phase2 = None
        
        logger.info("Trainer initialized")
    
    def _get_callbacks(self, phase: int = 1) -> list:
        """
        Get training callbacks for a specific phase.
        
        Args:
            phase: Training phase (1 or 2)
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.paths["model_dir"] / f"best_model_phase{phase}.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=Config.MIN_LR,
            verbose=1
        ))
        
        # CSV logging
        log_dir = Config.PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"training_phase{phase}_{timestamp}.csv"
        
        callbacks.append(CSVLogger(
            str(csv_path),
            separator=',',
            append=False
        ))
        
        return callbacks
    
    def train_phase1(
        self,
        train_generator,
        val_generator,
        epochs: int = Config.EPOCHS_PHASE1,
        class_weights: Dict[int, float] = None
    ) -> dict:
        """
        Phase 1: Train with frozen base layers.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history
        """
        logger.info("=" * 50)
        logger.info("PHASE 1: Transfer Learning (Frozen Base)")
        logger.info("=" * 50)
        
        # Compute class weights if not provided
        if class_weights is None:
            class_weights = get_class_weights(train_generator)
        
        # Get callbacks
        callbacks = self._get_callbacks(phase=1)
        
        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history_phase1 = history.history
        
        # Log results
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info(f"\nPhase 1 Complete!")
        logger.info(f"  Final val_accuracy: {final_val_acc:.4f}")
        logger.info(f"  Final val_loss: {final_val_loss:.4f}")
        
        return history.history
    
    def train_phase2(
        self,
        train_generator,
        val_generator,
        epochs: int = Config.EPOCHS_PHASE2,
        class_weights: Dict[int, float] = None,
        unfreeze_layers: int = Config.UNFREEZE_LAYERS
    ) -> dict:
        """
        Phase 2: Fine-tune with unfrozen top layers.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            class_weights: Optional class weights
            unfreeze_layers: Number of layers to unfreeze
            
        Returns:
            Training history
        """
        logger.info("=" * 50)
        logger.info("PHASE 2: Fine-Tuning (Unfrozen Top Layers)")
        logger.info("=" * 50)
        
        # Unfreeze model
        self.model = unfreeze_model(
            self.model,
            unfreeze_layers=unfreeze_layers,
            learning_rate=Config.LEARNING_RATE_PHASE2
        )
        
        # Compute class weights if not provided
        if class_weights is None:
            class_weights = get_class_weights(train_generator)
        
        # Get callbacks
        callbacks = self._get_callbacks(phase=2)
        
        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history_phase2 = history.history
        
        # Log results
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info(f"\nPhase 2 Complete!")
        logger.info(f"  Final val_accuracy: {final_val_acc:.4f}")
        logger.info(f"  Final val_loss: {final_val_loss:.4f}")
        
        return history.history
    
    def train_full_pipeline(
        self,
        train_generator,
        val_generator,
        class_weights: Dict[int, float] = None
    ) -> Tuple[dict, dict]:
        """
        Run complete two-phase training pipeline.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            class_weights: Optional class weights
            
        Returns:
            Tuple of (phase1_history, phase2_history)
        """
        logger.info("Starting full training pipeline...")
        
        # Phase 1
        history1 = self.train_phase1(
            train_generator,
            val_generator,
            class_weights=class_weights
        )
        
        # Phase 2
        history2 = self.train_phase2(
            train_generator,
            val_generator,
            class_weights=class_weights
        )
        
        # Save final model
        save_model(self.model, str(self.paths["model_path"]))
        
        # Save training history
        self._save_history()
        
        logger.info("Training pipeline complete!")
        
        return history1, history2
    
    def _save_history(self) -> None:
        """Save training history to JSON file."""
        history = {
            "phase1": self.history_phase1,
            "phase2": self.history_phase2,
            "config": {
                "epochs_phase1": Config.EPOCHS_PHASE1,
                "epochs_phase2": Config.EPOCHS_PHASE2,
                "learning_rate_phase1": Config.LEARNING_RATE_PHASE1,
                "learning_rate_phase2": Config.LEARNING_RATE_PHASE2,
                "batch_size": Config.BATCH_SIZE,
                "dropout_rate": Config.DROPOUT_RATE
            }
        }
        
        # Save to reports directory
        reports_dir = Config.REPORTS_DIR
        reports_dir.mkdir(exist_ok=True)
        
        history_path = reports_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to: {history_path}")
    
    def get_combined_history(self) -> dict:
        """
        Combine histories from both phases.
        
        Returns:
            Combined training history
        """
        if self.history_phase1 is None:
            return self.history_phase2 or {}
        
        if self.history_phase2 is None:
            return self.history_phase1
        
        combined = {}
        for key in self.history_phase1.keys():
            combined[key] = self.history_phase1[key] + self.history_phase2.get(key, [])
        
        return combined


if __name__ == "__main__":
    print("Trainer Module Test")
    print("=" * 40)
    
    trainer = Trainer()
    print(f"Model name: {trainer.model.name}")
    print(f"Model params: {trainer.model.count_params():,}")
    print("Module loaded successfully!")
