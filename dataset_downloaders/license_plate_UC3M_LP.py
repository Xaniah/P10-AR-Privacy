# Source: https://github.com/ramajoballester/UC3M-LP?tab=readme-ov-file
import sys
import os

# Add project root to path so package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from ultralytics import SETTINGS
from ultralytics.utils.downloads import download

from utils import transform_dataset

def download_license_plate_uc3m_lp():
  dataset_dir = Path(SETTINGS["datasets_dir"]) / "UC3M-LP"
  dataset_url = "https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1"

  if (dataset_dir).exists():
    print(f"UC3M-LP Dataset already exists at {dataset_dir}, skipping download.")
    return
  
  download(url=dataset_url, dir=dataset_dir, threads=3, delete=True)
  transform_dataset(dataset_dir, lp_size=2080)


if __name__ == "__main__":
  download_license_plate_uc3m_lp()