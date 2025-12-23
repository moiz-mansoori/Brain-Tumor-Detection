"""
Tests for Brain Cropping Preprocessing
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.preprocessing.brain_cropper import BrainCropper

def test_brain_cropper_init():
    """Test initialization of BrainCropper."""
    target_size = (100, 100)
    cropper = BrainCropper(target_size=target_size)
    assert cropper.target_size == target_size
    assert cropper.processed_count == 0


def test_crop_brain_region_shapes():
    """Test that cropping returns the correct shape."""
    target_size = (224, 224)
    cropper = BrainCropper(target_size=target_size)
    
    # Create a dummy image (black with a white circle in the middle representing the brain)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(image, (150, 150), 50, (255, 255, 255), -1)
    
    cropped, success = cropper.crop_brain_region(image)
    
    assert success is True
    assert cropped.shape == (224, 224, 3)


def test_crop_brain_region_fallback():
    """Test fallback when no brain region is found."""
    target_size = (224, 224)
    cropper = BrainCropper(target_size=target_size)
    
    # Create a completely black image (no contours)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    cropped, success = cropper.crop_brain_region(image)
    
    assert success is False
    assert cropped.shape == (224, 224, 3) # Should still resize to target


def test_process_invalid_image_path():
    """Test processing an invalid image path."""
    cropper = BrainCropper()
    processed, success = cropper.process_image("non_existent_path.jpg")
    
    assert processed is None
    assert success is False
    assert "non_existent_path.jpg" in cropper.failed_images
