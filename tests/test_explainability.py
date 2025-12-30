"""
Tests for Grad-CAM Explainability Module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


class TestGradCAMFunctions:
    """Tests for Grad-CAM implementation."""
    
    def test_overlay_heatmap_output_shape(self):
        """Test that overlay_heatmap returns correct shape."""
        from src.utils.explainability import overlay_heatmap
        
        # Create dummy image and heatmap
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(14, 14).astype(np.float32)  # Typical VGG-16 conv5 output
        
        result = overlay_heatmap(img, heatmap, alpha=0.5)
        
        assert result.shape == img.shape, f"Expected shape {img.shape}, got {result.shape}"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    
    def test_overlay_heatmap_alpha_values(self):
        """Test overlay with different alpha values."""
        from src.utils.explainability import overlay_heatmap
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        heatmap = np.ones((7, 7), dtype=np.float32)
        
        # Test different alpha values
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = overlay_heatmap(img, heatmap, alpha=alpha)
            assert result.shape == img.shape


class TestGradCAMWithModel:
    """Tests that require the actual trained model."""
    
    @pytest.fixture
    def model(self):
        """Load the trained model for testing."""
        import tensorflow as tf
        model_path = Path(__file__).parent.parent / "saved_models" / "vgg16_brain_tumor_4class.h5"
        
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")
        
        return tf.keras.models.load_model(str(model_path))
    
    @pytest.fixture
    def sample_input(self):
        """Create a sample preprocessed input image."""
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
        # Create a random image
        img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        return preprocess_input(img.astype(np.float32))
    
    def test_generate_gradcam_returns_heatmap(self, model, sample_input):
        """Test that generate_gradcam returns a valid heatmap."""
        from src.utils.explainability import generate_gradcam
        
        heatmap = generate_gradcam(model, sample_input)
        
        # Check heatmap properties
        assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
        assert heatmap.ndim == 2, f"Heatmap should be 2D, got {heatmap.ndim}D"
        assert heatmap.min() >= 0, "Heatmap values should be >= 0"
        assert heatmap.max() <= 1, "Heatmap values should be <= 1"
    
    def test_generate_gradcam_with_pred_index(self, model, sample_input):
        """Test Grad-CAM with a specified prediction index."""
        from src.utils.explainability import generate_gradcam
        
        # Test with each class index
        for pred_idx in range(4):
            heatmap = generate_gradcam(model, sample_input, pred_index=pred_idx)
            assert heatmap is not None, f"Heatmap should not be None for class {pred_idx}"
            assert heatmap.ndim == 2, f"Heatmap should be 2D for class {pred_idx}"
    
    def test_end_to_end_gradcam_visualization(self, model, sample_input):
        """Test complete Grad-CAM workflow."""
        from src.utils.explainability import generate_gradcam, overlay_heatmap
        
        # Generate heatmap
        heatmap = generate_gradcam(model, sample_input)
        
        # Create overlay
        original_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = overlay_heatmap(original_img, heatmap, alpha=0.5)
        
        # Verify output
        assert overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {overlay.shape}"
        assert overlay.dtype == np.uint8, f"Overlay dtype should be uint8, got {overlay.dtype}"


class TestInputValidation:
    """Test input validation in Grad-CAM functions."""
    
    def test_generate_gradcam_invalid_input_dims(self):
        """Test that invalid input dimensions raise ValueError."""
        from src.utils.explainability import generate_gradcam
        import tensorflow as tf
        
        # Create a dummy model (won't be used due to early validation)
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
        
        # Test with wrong dimensions
        invalid_input = np.random.rand(224, 224, 3)  # 3D instead of 4D
        
        with pytest.raises(ValueError, match="4 dims"):
            generate_gradcam(model, invalid_input)
