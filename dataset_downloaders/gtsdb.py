# Source: https://www.kaggle.com/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset/data
# Used for traffic signs

from pathlib import Path

import kagglehub
from ultralytics import SETTINGS

from dataset_postprocessors import remap_class_ids
from utils.dataset_config_utils import get_dataset_class_id_by_name


def download_gtsdb():
  dataset_dir = Path(SETTINGS["datasets_dir"]) / "GTSDB"

  if dataset_dir.exists():
    print(f"GTSDB Dataset already exists at {dataset_dir}, skipping download.")
    return

  kagglehub.dataset_download("icebearogo/german-traffic-sign-detection-gtsdb-dataset", output_dir=dataset_dir)

  traffic_sign_class_id = get_dataset_class_id_by_name("Traffic sign")
  remap_class_ids(dataset_dir / "GTSDB_Train_and_Test/Train/labels", traffic_sign_class_id)
  remap_class_ids(dataset_dir / "GTSDB_Train_and_Test/Test/labels", traffic_sign_class_id)

if __name__ == "__main__":
  download_gtsdb()