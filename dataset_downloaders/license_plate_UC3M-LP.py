# WIP
# Source: https://github.com/ramajoballester/UC3M-LP?tab=readme-ov-file
from pathlib import Path

from ultralytics import SETTINGS
from ultralytics.utils.downloads import download

def download_license_plate_uc3m_lp():
  dataset_dir = Path(SETTINGS["datasets_dir"]) / "UC3M-LP"
  dataset_url = "https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1"

  download(url=dataset_url, dir=dataset_dir, threads=3, delete=True)


if __name__ == "__main__":
  download_license_plate_uc3m_lp()