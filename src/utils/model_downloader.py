"""
Model Downloader - Downloads the trained model from Hugging Face Hub.

This enables deployment on platforms like Streamlit Cloud where large model files 
cannot be committed to GitHub (100MB limit).
"""

import os
from pathlib import Path
from typing import Optional


# Configuration - Update this after uploading to Hugging Face
HUGGINGFACE_REPO_ID = "moiz-mansoori/brain-tumor-vgg16"  # Your HF username/repo-name
MODEL_FILENAME = "vgg16_brain_tumor_4class.h5"


def get_model_path() -> Path:
    """Get the local path where the model should be stored."""
    return Path(__file__).parent.parent.parent / "saved_models" / MODEL_FILENAME


def download_model_from_hf(
    repo_id: str = HUGGINGFACE_REPO_ID,
    filename: str = MODEL_FILENAME,
    force_download: bool = False
) -> Optional[Path]:
    """
    Download the trained model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (username/repo-name)
        filename: Name of the model file in the repository
        force_download: If True, re-download even if file exists locally
        
    Returns:
        Path to the downloaded model file, or None if download fails
    """
    local_path = get_model_path()
    
    # Check if model already exists locally
    if local_path.exists() and not force_download:
        print(f"✅ Model already exists at: {local_path}")
        return local_path
    
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading model from Hugging Face: {repo_id}/{filename}")
        
        # Download to cache first, then we'll use directly from cache
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(local_path.parent),
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully!")
        return Path(downloaded_path)
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def ensure_model_available() -> Optional[Path]:
    """
    Ensure the model is available locally - download if necessary.
    
    This is the main function to call from the Streamlit app.
    
    Returns:
        Path to the model file, or None if unavailable
    """
    local_path = get_model_path()
    
    if local_path.exists():
        return local_path
    
    # Try to download from Hugging Face
    return download_model_from_hf()


if __name__ == "__main__":
    # Test the downloader
    model_path = ensure_model_available()
    if model_path:
        print(f"Model ready at: {model_path}")
    else:
        print("Model not available")
