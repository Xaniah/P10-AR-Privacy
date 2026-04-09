import os
import argparse

from pathlib import Path
from ultralytics import SETTINGS, YOLO

parser = argparse.ArgumentParser(description="Pseudo-label Open Images v7 dataset using YOLO model")
parser.add_argument("-m", "--model", type=str, help="Path to YOLO model")
parser.add_argument("-i", "--id", type=int, help="Class ID (check the dataset-config.yaml for the mapping between IDs and names)")
args = parser.parse_args()

dataset_dir = Path(SETTINGS["datasets_dir"]) / "open-images-v7"

if not os.path.exists(dataset_dir):
  print("Open Images v7 dataset not found, please run dataset_downloaders/open_images_v7.py or train.py to download it.")

model = YOLO(args.model)

def pseudo_label_split(split):
  model.predict(
    source=str(dataset_dir / "images" / split),
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.4,
    project=str(dataset_dir / "pseudo_labels" / split),
    name="",
    exist_ok=True,
  )

for split in "train", "val":
  pseudo_label_split(split)
