"""
Reproducibility Module

Sets random seeds for all libraries to ensure reproducible results.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow (import here to avoid loading if not needed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Additional TensorFlow determinism settings
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
    except ImportError:
        pass
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[SEED] Random seed set to {seed} for reproducibility")


def get_seed() -> int:
    """
    Get the configured random seed from Config.
    
    Returns:
        Random seed value
    """
    from .config import Config
    return Config.RANDOM_SEED


if __name__ == "__main__":
    set_seed(42)
    print("Seed module test passed!")
