"""
Data Splitter Module

Handles train/validation/test splitting with stratification.
Ensures no data leakage by splitting BEFORE preprocessing.
"""

import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import random
from collections import Counter

from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.seed import set_seed


logger = get_logger(__name__)


class DataSplitter:
    """
    Stratified data splitter for binary classification.
    
    Ensures:
    - Class ratios are preserved in each split
    - No data leakage (split before preprocessing)
    - Reproducible splits via seeding
    """
    
    def __init__(
        self,
        train_ratio: float = Config.TRAIN_RATIO,
        val_ratio: float = Config.VAL_RATIO,
        test_ratio: float = Config.TEST_RATIO,
        seed: int = Config.RANDOM_SEED
    ):
        """
        Initialize Data Splitter.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from a directory."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", 
                     ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"}
        
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        
        return sorted(files)  # Sort for reproducibility
    
    def _stratified_split(
        self,
        files: List[Path]
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split files into train/val/test sets.
        
        Args:
            files: List of file paths
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_files = shuffled[:train_end]
        val_files = shuffled[train_end:val_end]
        test_files = shuffled[val_end:]
        
        return train_files, val_files, test_files
    
    def copy_raw_data(
        self,
        source_dir: Path = None,
        dest_dir: Path = None
    ) -> Dict[str, int]:
        """
        Copy raw data from original dataset to data/raw/ directory.
        
        Args:
            source_dir: Source directory (default: Dataset/)
            dest_dir: Destination directory (default: data/raw/)
            
        Returns:
            Dictionary with copy statistics
        """
        source_dir = source_dir or Config.ORIGINAL_DATASET_DIR
        dest_dir = dest_dir or Config.RAW_DATA_DIR
        
        source_dir = Path(source_dir)
        dest_dir = Path(dest_dir)
        
        stats = {"yes": 0, "no": 0}
        
        for class_name in ["yes", "no"]:
            src_class_dir = source_dir / class_name
            dst_class_dir = dest_dir / class_name
            
            if not src_class_dir.exists():
                logger.warning(f"Source directory not found: {src_class_dir}")
                continue
            
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            files = self._get_image_files(src_class_dir)
            
            for file_path in files:
                dst_path = dst_class_dir / file_path.name
                if not dst_path.exists():
                    shutil.copy2(file_path, dst_path)
                stats[class_name] += 1
            
            logger.info(f"Copied {stats[class_name]} images to {dst_class_dir}")
        
        return stats
    
    def split_data(
        self,
        source_dir: Path = None,
        output_dir: Path = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Split data into train/val/test sets with stratification.
        
        Args:
            source_dir: Source directory with yes/no subdirectories
            output_dir: Output directory for splits
            
        Returns:
            Dictionary with split statistics
        """
        source_dir = source_dir or Config.RAW_DATA_DIR
        output_dir = output_dir or Config.SPLITS_DIR
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        stats = {
            "train": {"yes": 0, "no": 0},
            "val": {"yes": 0, "no": 0},
            "test": {"yes": 0, "no": 0}
        }
        
        for class_name in ["yes", "no"]:
            class_dir = source_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            files = self._get_image_files(class_dir)
            logger.info(f"Found {len(files)} images in class '{class_name}'")
            
            # Split
            train_files, val_files, test_files = self._stratified_split(files)
            
            # Copy to respective directories
            splits = [
                ("train", train_files),
                ("val", val_files),
                ("test", test_files)
            ]
            
            for split_name, split_files in splits:
                split_class_dir = output_dir / split_name / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for file_path in split_files:
                    dst_path = split_class_dir / file_path.name
                    shutil.copy2(file_path, dst_path)
                    stats[split_name][class_name] += 1
        
        # Log summary
        logger.info("\n" + "=" * 50)
        logger.info("DATA SPLIT SUMMARY")
        logger.info("=" * 50)
        
        for split_name in ["train", "val", "test"]:
            total = stats[split_name]["yes"] + stats[split_name]["no"]
            logger.info(f"{split_name.upper():6}: "
                       f"Yes={stats[split_name]['yes']:3}, "
                       f"No={stats[split_name]['no']:3}, "
                       f"Total={total:3}")
        
        return stats
    
    def get_class_distribution(self, directory: Path) -> Dict[str, int]:
        """
        Get class distribution in a directory.
        
        Args:
            directory: Directory with yes/no subdirectories
            
        Returns:
            Dictionary with class counts
        """
        directory = Path(directory)
        distribution = {}
        
        for class_name in ["yes", "no"]:
            class_dir = directory / class_name
            if class_dir.exists():
                files = self._get_image_files(class_dir)
                distribution[class_name] = len(files)
            else:
                distribution[class_name] = 0
        
        return distribution
    
    def compute_class_weights(self, directory: Path = None) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.
        
        Args:
            directory: Directory with training data
            
        Returns:
            Dictionary mapping class index to weight
        """
        directory = directory or (Config.SPLITS_DIR / "train")
        distribution = self.get_class_distribution(directory)
        
        total = sum(distribution.values())
        n_classes = len(distribution)
        
        # Compute weights: total / (n_classes * count)
        weights = {}
        for class_name, count in distribution.items():
            class_idx = Config.CLASS_LABELS[class_name]
            weights[class_idx] = total / (n_classes * max(1, count))
        
        logger.info(f"Class weights: {weights}")
        return weights


def prepare_multiclass_data() -> Dict:
    """
    Prepare multi-class dataset (glioma, meningioma, notumor, pituitary).
    
    The Kaggle dataset already has Training/Testing splits.
    We create a validation set by taking 15% from training.
    
    Returns:
        Dictionary with all statistics
    """
    logger.info("Starting multi-class data preparation...")
    
    # Create directories
    Config.create_directories()
    
    stats = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    # Set seed for reproducibility
    random.seed(Config.RANDOM_SEED)
    
    val_ratio = 0.15  # Take 15% of training for validation
    
    for class_name in Config.CLASSES:
        # Source directories
        train_src = Config.TRAINING_DIR / class_name
        test_src = Config.TESTING_DIR / class_name
        
        # Destination directories
        train_dst = Config.SPLITS_DIR / "train" / class_name
        val_dst = Config.SPLITS_DIR / "val" / class_name
        test_dst = Config.SPLITS_DIR / "test" / class_name
        
        # Create directories
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)
        
        # Get training files
        train_files = list(train_src.glob("*.[jJ][pP][gG]")) + \
                      list(train_src.glob("*.[pP][nN][gG]")) + \
                      list(train_src.glob("*.[jJ][pP][eE][gG]"))
        
        random.shuffle(train_files)
        
        # Split training into train + val
        n_val = int(len(train_files) * val_ratio)
        val_files = train_files[:n_val]
        train_files = train_files[n_val:]
        
        # Copy training files
        for f in train_files:
            shutil.copy2(f, train_dst / f.name)
        stats["train"][class_name] = len(train_files)
        
        # Copy validation files
        for f in val_files:
            shutil.copy2(f, val_dst / f.name)
        stats["val"][class_name] = len(val_files)
        
        # Get and copy test files
        test_files = list(test_src.glob("*.[jJ][pP][gG]")) + \
                     list(test_src.glob("*.[pP][nN][gG]")) + \
                     list(test_src.glob("*.[jJ][pP][eE][gG]"))
        
        for f in test_files:
            shutil.copy2(f, test_dst / f.name)
        stats["test"][class_name] = len(test_files)
        
        logger.info(f"{class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-CLASS DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        logger.info(f"{split.upper():5}: {total} images")
    
    return stats


def prepare_data() -> Dict:
    """
    Complete data preparation pipeline.
    Automatically detects which dataset structure to use.
    
    Returns:
        Dictionary with all statistics
    """
    # Check if new multi-class dataset exists
    if Config.TRAINING_DIR.exists() and Config.TESTING_DIR.exists():
        logger.info("Detected multi-class Kaggle dataset structure")
        return prepare_multiclass_data()
    
    # Fall back to old binary dataset
    logger.info("Using binary dataset structure")
    logger.info("Starting data preparation...")
    
    # Create directories
    Config.create_directories()
    
    splitter = DataSplitter()
    
    # Step 1: Copy raw data
    logger.info("\nStep 1: Copying raw data...")
    copy_stats = splitter.copy_raw_data()
    
    # Step 2: Split data
    logger.info("\nStep 2: Splitting data...")
    split_stats = splitter.split_data()
    
    # Step 3: Compute class weights
    logger.info("\nStep 3: Computing class weights...")
    class_weights = splitter.compute_class_weights()
    
    return {
        "copy_stats": copy_stats,
        "split_stats": split_stats,
        "class_weights": class_weights
    }


if __name__ == "__main__":
    print("Data Splitter Module Test")
    print("=" * 40)
    
    splitter = DataSplitter()
    print(f"Train ratio: {splitter.train_ratio}")
    print(f"Val ratio: {splitter.val_ratio}")
    print(f"Test ratio: {splitter.test_ratio}")
    print("Module loaded successfully!")
