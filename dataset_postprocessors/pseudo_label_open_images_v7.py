import os
import argparse
import re

from tqdm import tqdm
from pathlib import Path
import shutil
from ultralytics import SETTINGS, YOLO

parser = argparse.ArgumentParser(description="Pseudo-label Open Images v7 dataset using YOLO model")
parser.add_argument("-m", "--model", type=str, help="Path to YOLO model")
parser.add_argument("-mc", "--model-confidence", type=float, default=0.5, help="Confidence threshold for YOLO predictions (default: 0.5)")
parser.add_argument("-mi", "--model-image-size", type=int, default=1280, help="Image size for YOLO predictions (default: 1280)")
parser.add_argument("-i", "--id", type=int, help="Class ID for the pseudo-labels (check the dataset-config.yaml for the mapping between IDs and names)")
parser.add_argument(
  "-c",
  "--config",
  choices=["label", "merge", "full"],
  help=(
    "Config to use. Options:\n"
    "  label     : Pseudo-label images, save detections in 'datasets_dir / pseudo_labels' folder. If the folder already exists, it will be removed.\n"
    "  merge     : Merge pseudo-labelled images with dataset (Removes the pseudo-labels folder)\n"
    "  full      : Perform both steps (label + merge)\n\n"
    "Notes:\n"
    "  - The order to execute this script is: label, then merge. The 'full' config is just a shortcut to execute both steps in one go.\n"
    "  - The pseudo-labels will be saved with the specified class ID (-i / --id option)\n"
    "  - 'merge' config assumes that the 'label' config has already been run and the pseudo-labels are available in the 'datasets_dir / pseudo_labels' folder.\n"
  )
)
args = parser.parse_args()

dataset_dir = Path(SETTINGS["datasets_dir"]) / "open-images-v7"
dataset_doesnt_exist_msg = "Open Images v7 dataset not found, please run dataset_downloaders/open_images_v7.py or train.py to download it."

def label():
  if not os.path.exists(dataset_dir):
    print(dataset_doesnt_exist_msg)
    return

  pseudo_labels_dir = dataset_dir / "pseudo_labels"

  if os.path.exists(pseudo_labels_dir):
    shutil.rmtree(pseudo_labels_dir)

  model = YOLO(args.model)

  detections = 0

  for split in "train", "val":
    results = model.predict(
      source=str(dataset_dir / "images" / split),
      save=True,
      save_txt=True,
      save_conf=True,
      conf=args.model_confidence,
      project=str(pseudo_labels_dir / split),
      imgsz=args.model_image_size
    )
    
    for r in results:
      detections += len(r.boxes)

  print(f"Total detections: {detections}")


def merge():
  if not os.path.exists(dataset_dir):
    print(dataset_doesnt_exist_msg)
    return

  pseudo_labels_dir = dataset_dir / "pseudo_labels"

  for split in "train", "val":
    split_dir = pseudo_labels_dir / split / "predict/labels"
    if not os.path.exists(split_dir):
      print(f"Pseudo-labels for {split} split not found, please run the 'label' config first.")
      return

    print(f"Merging pseudo-labels for {split} split...")
    for file in tqdm(os.listdir(split_dir)):
      dst_img = dataset_dir / "labels" / split / file
      if dst_img.exists():
        src_img_file = open(split_dir / file, "r")
        src_img_content = src_img_file.read()
        remapped_content = re.sub(r'^0\s', f'{args.id} ', src_img_content, flags=re.MULTILINE)

        with open(dst_img, "a") as dst_file:
          dst_file.write("\n" + remapped_content)
        src_img_file.close()

  shutil.rmtree(pseudo_labels_dir) # Cleanup pseudo-labels after merging

def full():
  print("Starting full pseudo-labelling and merging process...")
  label()
  merge()

if __name__ == "__main__":
  if args.config == "label":
    label()
  elif args.config == "merge":
    merge()
  elif args.config == "full":
    full()