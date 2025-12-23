import shutil
from pathlib import Path
import os
import sys

# Add project root to path to import config if needed, 
# but for this simple script we'll just determine root relatively.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_FILENAME = BASE_DIR / "dataset" # shutil.make_archive adds .zip extension automatically

def create_dataset_archive():
    """Create a zip archive of the Training and Testing datasets."""
    
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        sys.exit(1)
        
    print(f"Creating dataset archive from {DATASET_DIR}...")
    
    # We want the zip to contain 'Training' and 'Testing' at the root level.
    # So we zip the CONTENTS of Dataset, not the Dataset folder itself.
    
    try:
        # shutil.make_archive(base_name, format, root_dir, base_dir)
        # root_dir is the directory which will be the root of the archive
        archive_path = shutil.make_archive(
            base_name=str(OUTPUT_FILENAME),
            format='zip',
            root_dir=str(DATASET_DIR),
            base_dir=None # ZIP everything inside root_dir
        )
        print(f"✅ Success! Dataset archive created at: {archive_path}")
        print(f"Next step: Upload '{Path(archive_path).name}' to your Google Drive.")
        
    except Exception as e:
        print(f"❌ Error creating archive: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_dataset_archive()
