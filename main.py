#!/usr/bin/env python
"""
Brain Tumor Detection - Command Line Interface

Complete pipeline orchestration for:
- Data preparation (copy, split)
- Preprocessing (brain cropping)
- Training (two-phase VGG-16)
- Evaluation (metrics, visualization)
- Prediction (single image inference)

Usage:
    python main.py prepare      # Copy data and create splits
    python main.py preprocess   # Apply brain cropping
    python main.py train        # Run training pipeline
    python main.py evaluate     # Evaluate on test set
    python main.py predict <image_path>  # Single image prediction
    python main.py app          # Launch Streamlit app
    python main.py pipeline     # Run complete pipeline
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


logger = get_logger(__name__)


def cmd_prepare(args):
    """Prepare data: copy from Dataset/ and create train/val/test splits."""
    print("\n" + "=" * 60)
    print("📂 DATA PREPARATION")
    print("=" * 60)
    
    from src.preprocessing.data_splitter import DataSplitter, prepare_data
    
    # Create directories
    Config.create_directories()
    print(f"✅ Created directory structure")
    
    # Run complete preparation
    stats = prepare_data()
    
    print("\n📊 Data Preparation Summary:")
    print(f"  Raw data copied: {stats.get('copy_stats', {}).get('total', 'N/A')} files")
    print(f"  Train split: {stats.get('split_stats', {}).get('train', 'N/A')} files")
    print(f"  Val split: {stats.get('split_stats', {}).get('val', 'N/A')} files")
    print(f"  Test split: {stats.get('split_stats', {}).get('test', 'N/A')} files")
    
    print("\n✅ Data preparation complete!")
    return stats


def cmd_preprocess(args):
    """Apply brain cropping to all split images."""
    print("\n" + "=" * 60)
    print("🔬 BRAIN CROPPING PREPROCESSING")
    print("=" * 60)
    
    from src.preprocessing.brain_cropper import BrainCropper
    
    cropper = BrainCropper()
    
    # Process each split
    splits = ['train', 'val', 'test']
    total_stats = {}
    
    for split in splits:
        print(f"\n📁 Processing {split} split...")
        
        input_dir = Config.SPLITS_DIR / split
        output_dir = Config.PROCESSED_DIR / split
        
        if not input_dir.exists():
            print(f"  ⚠️ {input_dir} not found. Run 'prepare' first.")
            continue
        
        stats = cropper.process_directory(input_dir, output_dir)
        total_stats[split] = stats
        
        print(f"  ✅ Processed: {stats.get('success', 0)}/{stats.get('total', 0)} images")
        if stats.get('failed', 0) > 0:
            print(f"  ⚠️ Failed: {stats['failed']} images")
    
    print("\n✅ Brain cropping complete!")
    return total_stats


def cmd_train(args):
    """Run two-phase training pipeline."""
    print("\n" + "=" * 60)
    print("🧠 TRAINING PIPELINE")
    print("=" * 60)
    
    from src.training.trainer import Trainer
    from src.preprocessing.data_loader import create_generators, get_class_weights
    
    # Set seed for reproducibility
    set_seed(Config.RANDOM_SEED)
    
    # Check for processed data
    if not (Config.PROCESSED_DIR / "train").exists():
        print("❌ Processed data not found. Run 'prepare' and 'preprocess' first.")
        return None
    
    # Create data generators
    print("\n📊 Loading data generators...")
    train_gen, val_gen, test_gen = create_generators()
    
    # Compute class weights
    class_weights = get_class_weights(train_gen)
    print(f"  Class weights: {class_weights}")
    
    # Initialize trainer
    print("\n🏗️ Building model...")
    trainer = Trainer()
    
    # Run full training pipeline
    print("\n🚀 Starting training...")
    history1, history2 = trainer.train_full_pipeline(
        train_gen, val_gen, class_weights
    )
    
    print("\n✅ Training complete!")
    print(f"  Model saved to: {Config.MODEL_PATH}")
    
    return trainer


def cmd_evaluate(args):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION")
    print("=" * 60)
    
    from src.evaluation.metrics import (
        evaluate_model, 
        generate_evaluation_report,
        plot_confusion_matrix,
        plot_roc_curve
    )
    from src.preprocessing.data_loader import create_generators
    from src.models.vgg16_model import load_model
    
    # Check for model
    if not Config.MODEL_PATH.exists():
        print(f"❌ Model not found at: {Config.MODEL_PATH}")
        print("  Train the model first using 'train' command.")
        return None
    
    # Load model
    print("\n🔄 Loading model...")
    model = load_model()
    
    # Load test data
    print("📂 Loading test data...")
    _, _, test_gen = create_generators()
    
    # Evaluate
    print("\n🔍 Evaluating model...")
    results = evaluate_model(model, test_gen)
    
    # Generate report
    report_path = Config.REPORTS_DIR / "evaluation_report.txt"
    generate_evaluation_report(results, str(report_path))
    print(f"\n📄 Report saved to: {report_path}")
    
    # Generate plots
    print("📉 Generating plots...")
    cm_path = Config.REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(model, test_gen, save_path=str(cm_path))
    
    roc_path = Config.REPORTS_DIR / "roc_curve.png"
    plot_roc_curve(model, test_gen, save_path=str(roc_path))
    
    print("\n✅ Evaluation complete!")
    return results


def cmd_predict(args):
    """Predict on a single image (4-class: glioma, meningioma, notumor, pituitary)."""
    print("\n" + "=" * 60)
    print("🔮 SINGLE IMAGE PREDICTION (4-CLASS)")
    print("=" * 60)
    
    import cv2
    import numpy as np
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from src.preprocessing.brain_cropper import crop_brain_from_image
    from src.models.vgg16_model import load_model
    
    image_path = args.image
    
    # Validate image path
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Check for model
    if not Config.MODEL_PATH.exists():
        print(f"❌ Model not found at: {Config.MODEL_PATH}")
        return None
    
    print(f"\n📷 Input image: {image_path}")
    
    # Load and preprocess image
    print("🔄 Processing image...")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    
    # Apply brain cropping
    cropped = crop_brain_from_image(image)
    print("  ✅ Brain region extracted")
    
    # Prepare for model
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    
    # Load model and predict
    print("🧠 Loading model...")
    model = load_model()
    
    print("🔮 Making prediction...")
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get results
    predicted_idx = int(np.argmax(predictions))
    predicted_class = Config.CLASSES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    # Display result
    print("\n" + "-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    
    display_name = Config.CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)
    
    if predicted_class == "notumor":
        print(f"  ✅ {display_name.upper()}")
    else:
        print(f"  ⚠️  {display_name.upper()}")
    
    print(f"  Confidence: {confidence * 100:.1f}%")
    print("-" * 40)
    
    # Show all probabilities
    print("\nAll class probabilities:")
    for i, cls in enumerate(Config.CLASSES):
        prob = float(predictions[i])
        marker = "→" if i == predicted_idx else " "
        print(f"  {marker} {Config.CLASS_DISPLAY_NAMES.get(cls, cls)}: {prob*100:.1f}%")
    
    return {
        "image": image_path,
        "prediction": predicted_class,
        "display_name": display_name,
        "confidence": confidence,
        "all_probabilities": {cls: float(predictions[i]) for i, cls in enumerate(Config.CLASSES)}
    }


def cmd_app(args):
    """Launch Streamlit web application."""
    print("\n" + "=" * 60)
    print("🌐 LAUNCHING STREAMLIT APP")
    print("=" * 60)
    
    import subprocess
    
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ App not found at: {app_path}")
        return
    
    print(f"\n🚀 Starting server...")
    print(f"  App: {app_path}")
    print("\n  Press Ctrl+C to stop the server\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true"
    ])


def cmd_pipeline(args):
    """Run complete pipeline: prepare → preprocess → train → evaluate."""
    print("\n" + "=" * 60)
    print("🚀 COMPLETE PIPELINE EXECUTION")
    print("=" * 60)
    
    # Step 1: Prepare
    print("\n📌 Step 1/4: Preparing data...")
    cmd_prepare(args)
    
    # Step 2: Preprocess
    print("\n📌 Step 2/4: Preprocessing images...")
    cmd_preprocess(args)
    
    # Step 3: Train
    print("\n📌 Step 3/4: Training model...")
    cmd_train(args)
    
    # Step 4: Evaluate
    print("\n📌 Step 4/4: Evaluating model...")
    cmd_evaluate(args)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  Model saved to: {Config.MODEL_PATH}")
    print(f"  Run 'python main.py app' to launch the web interface")


def cmd_info(args):
    """Display project configuration."""
    Config.display()
    
    # Check data status
    print("\n📊 Data Status:")
    
    # Check for multi-class dataset (Training/Testing structure)
    if Config.TRAINING_DIR.exists() and Config.TESTING_DIR.exists():
        print("  Dataset type: Multi-class (4 classes)")
        print("\n  Training set:")
        train_total = 0
        for cls in Config.CLASSES:
            cls_dir = Config.TRAINING_DIR / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*")))
                print(f"    {cls}: {count} images")
                train_total += count
        print(f"    Total Training: {train_total}")

        print("\n  Testing set:")
        test_total = 0
        for cls in Config.CLASSES:
            cls_dir = Config.TESTING_DIR / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*")))
                print(f"    {cls}: {count} images")
                test_total += count
        print(f"    Total Testing: {test_total}")
        
        print(f"\n  Overall Total: {train_total + test_total} images")
    else:
        print(f"  ❌ Dataset not found at: {Config.ORIGINAL_DATASET_DIR}")
    
    # Check processed data
    print("\n  Processed data:")
    for split in ['train', 'val', 'test']:
        split_dir = Config.PROCESSED_DIR / split
        if split_dir.exists():
            total = sum(len(list((split_dir / cls).glob("*"))) 
                       for cls in Config.CLASSES 
                       if (split_dir / cls).exists())
            print(f"    {split.capitalize()}: {total} images")
        else:
            print(f"    {split.capitalize()}: Not prepared")
    
    # Check model
    print("\n🧠 Model Status:")
    if Config.MODEL_PATH.exists():
        size_mb = Config.MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"  Trained model: ✅ Found ({size_mb:.1f} MB)")
        print(f"  Path: {Config.MODEL_PATH}")
    else:
        print(f"  Trained model: ❌ Not found")
        print(f"  Expected path: {Config.MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection - CLI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py info              # Show project status
  python main.py prepare           # Copy and split data
  python main.py preprocess        # Apply brain cropping
  python main.py train             # Train the model
  python main.py evaluate          # Evaluate on test set
  python main.py predict image.jpg # Predict single image
  python main.py app               # Launch Streamlit app
  python main.py pipeline          # Run complete pipeline
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Display project configuration and status')
    parser_info.set_defaults(func=cmd_info)
    
    # Prepare command
    parser_prepare = subparsers.add_parser('prepare', help='Copy data and create train/val/test splits')
    parser_prepare.set_defaults(func=cmd_prepare)
    
    # Preprocess command
    parser_preprocess = subparsers.add_parser('preprocess', help='Apply brain cropping to all images')
    parser_preprocess.set_defaults(func=cmd_preprocess)
    
    # Train command
    parser_train = subparsers.add_parser('train', help='Run two-phase training pipeline')
    parser_train.set_defaults(func=cmd_train)
    
    # Evaluate command
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    parser_evaluate.set_defaults(func=cmd_evaluate)
    
    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Predict on a single image')
    parser_predict.add_argument('image', type=str, help='Path to MRI image file')
    parser_predict.set_defaults(func=cmd_predict)
    
    # App command
    parser_app = subparsers.add_parser('app', help='Launch Streamlit web application')
    parser_app.set_defaults(func=cmd_app)
    
    # Pipeline command
    parser_pipeline = subparsers.add_parser('pipeline', help='Run complete pipeline (prepare → preprocess → train → evaluate)')
    parser_pipeline.set_defaults(func=cmd_pipeline)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
