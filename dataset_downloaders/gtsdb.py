# WIP
# Source: https://www.kaggle.com/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset/data
# Used for traffic signs

from pathlib import Path

import kagglehub
from ultralytics import SETTINGS


def download_gtsdb():
  dataset_dir = Path(SETTINGS["datasets_dir"]) / "GTSDB"
  kagglehub.dataset_download("icebearogo/german-traffic-sign-detection-gtsdb-dataset", output_dir=dataset_dir)
