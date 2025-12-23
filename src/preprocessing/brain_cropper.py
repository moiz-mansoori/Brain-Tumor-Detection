"""
Brain Region Cropper Module

Automated Computer Vision pipeline for extracting brain regions from MRI images.
Uses contour detection to isolate the brain and remove background noise.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class BrainCropper:
    """
    Automated brain region extraction from MRI images.
    
    Pipeline:
    1. Convert to grayscale
    2. Apply Gaussian Blur
    3. Apply Otsu/adaptive thresholding
    4. Detect contours
    5. Select largest brain contour
    6. Crop bounding box with padding
    7. Resize to target size
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = Config.IMAGE_SIZE,
        blur_kernel: Tuple[int, int] = Config.BLUR_KERNEL_SIZE,
        threshold_method: str = Config.THRESHOLD_METHOD,
        min_contour_area: int = Config.MIN_CONTOUR_AREA,
        padding: int = Config.PADDING
    ):
        """
        Initialize Brain Cropper.
        
        Args:
            target_size: Output image size (width, height)
            blur_kernel: Gaussian blur kernel size
            threshold_method: "otsu" or "adaptive"
            min_contour_area: Minimum area to consider as brain region
            padding: Pixels to add around bounding box
        """
        self.target_size = target_size
        self.blur_kernel = blur_kernel
        self.threshold_method = threshold_method
        self.min_contour_area = min_contour_area
        self.padding = padding
        
        # Track statistics
        self.processed_count = 0
        self.fallback_count = 0
        self.failed_images: List[str] = []
    
    def crop_brain_region(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Extract brain region from MRI image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (cropped_image, success_flag)
        """
        original = image.copy()
        
        # Ensure we have a grayscale image for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Step 2: Apply thresholding
        if self.threshold_method == "otsu":
            _, thresh = cv2.threshold(
                blurred, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # adaptive
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Step 4: Find the largest valid contour (brain region)
        if not contours:
            logger.warning("No contours found, using original image")
            return self._resize_image(original), False
        
        # Filter contours by minimum area
        valid_contours = [
            c for c in contours 
            if cv2.contourArea(c) >= self.min_contour_area
        ]
        
        if not valid_contours:
            logger.warning("No valid contours found, using original image")
            return self._resize_image(original), False
        
        # Get the largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Step 5: Get bounding box with padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding (ensure we don't go out of bounds)
        height, width = gray.shape[:2]
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(width, x + w + self.padding)
        y2 = min(height, y + h + self.padding)
        
        # Step 6: Crop the region
        if len(original.shape) == 3:
            cropped = original[y1:y2, x1:x2]
        else:
            cropped = original[y1:y2, x1:x2]
        
        # Step 7: Resize to target size
        resized = self._resize_image(cropped)
        
        return resized, True
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(
            image, self.target_size, 
            interpolation=cv2.INTER_AREA
        )
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (processed_image, success_flag)
        """
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            self.failed_images.append(str(image_path))
            return None, False
        
        # Process
        processed, success = self.crop_brain_region(image)
        
        self.processed_count += 1
        if not success:
            self.fallback_count += 1
            self.failed_images.append(str(image_path))
        
        return processed, success
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        preserve_structure: bool = True
    ) -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            preserve_structure: If True, preserve subdirectory structure
            
        Returns:
            Dictionary with processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Reset statistics
        self.processed_count = 0
        self.fallback_count = 0
        self.failed_images = []
        
        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}
        
        # Process each class directory
        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            out_class_dir = output_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing class: {class_name}")
            
            for image_path in class_dir.iterdir():
                if image_path.suffix not in extensions:
                    continue
                
                processed, success = self.process_image(str(image_path))
                
                if processed is not None:
                    # Save processed image
                    output_path = out_class_dir / f"{image_path.stem}.jpg"
                    cv2.imwrite(str(output_path), processed)
        
        # Log statistics
        success_rate = ((self.processed_count - self.fallback_count) / 
                       max(1, self.processed_count) * 100)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total processed: {self.processed_count}")
        logger.info(f"Fallback to original: {self.fallback_count}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return {
            "total_processed": self.processed_count,
            "fallback_count": self.fallback_count,
            "success_rate": success_rate,
            "failed_images": self.failed_images
        }
    
    def visualize_pipeline(self, image_path: str) -> dict:
        """
        Visualize the complete pipeline for a single image.
        Useful for debugging and demonstration.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with all intermediate images
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Store all stages
        stages = {"original": image.copy()}
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stages["grayscale"] = gray
        
        # Blur
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        stages["blurred"] = blurred
        
        # Threshold
        _, thresh = cv2.threshold(
            blurred, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        stages["threshold"] = thresh
        
        # Contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours on original
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        stages["contours"] = contour_img
        
        # Final cropped
        cropped, success = self.crop_brain_region(image)
        stages["cropped"] = cropped
        stages["success"] = success
        
        return stages


def crop_brain_from_image(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to crop brain region from an image array.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Cropped and resized image
    """
    cropper = BrainCropper()
    cropped, _ = cropper.crop_brain_region(image)
    return cropped


if __name__ == "__main__":
    # Test the brain cropper
    from ..utils.config import Config
    
    print("Brain Cropper Module Test")
    print("=" * 40)
    
    cropper = BrainCropper()
    print(f"Target size: {cropper.target_size}")
    print(f"Threshold method: {cropper.threshold_method}")
    print("Module loaded successfully!")
