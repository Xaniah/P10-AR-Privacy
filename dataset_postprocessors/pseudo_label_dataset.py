import os
import argparse
import re

from tqdm import tqdm
from pathlib import Path
import shutil
from ultralytics import SETTINGS, YOLO

parser = argparse.ArgumentParser(description="Pseudo-label a dataset using the YOLO model")
parser.add_argument("-m", "--model", type=str, help="Path to YOLO model")
parser.add_argument("-mc", "--model-confidence", type=float, default=0.6, help="Confidence threshold for YOLO predictions (default: 0.6)")
parser.add_argument("-mi", "--model-image-size", type=int, default=1280, help="Image size for YOLO predictions (default: 1280)")
parser.add_argument("-i", "--id", type=int, help="Class ID for the pseudo-labels (check the dataset-config.yaml for the mapping between IDs and names)")
parser.add_argument("-ip", "--images-path", type=str, help="Path to the images folder inside the dataset (starts from the datasets_dir specified in the YOLO settings)")
parser.add_argument("-lp", "--labels-path", type=str, help="Path to the labels folder inside the dataset (starts from the datasets_dir specified in the YOLO settings)")
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

dataset_dir = Path(SETTINGS["datasets_dir"])
dataset_images_dir = dataset_dir / args.images_path
dataset_labels_dir = dataset_dir / args.labels_path
dataset_doesnt_exist_msg = "Dataset not found, please run the corresponding dataset downloader or train.py to download it (e.g. dataset_downloaders/open_images_v7.py)"

def label():
  if not os.path.exists(dataset_images_dir) or not os.path.exists(dataset_labels_dir):
    print(dataset_doesnt_exist_msg)
    return

  pseudo_labels_dir = dataset_dir / "pseudo_labels"

  if os.path.exists(pseudo_labels_dir):
    shutil.rmtree(pseudo_labels_dir)

  model = YOLO(args.model)

  detections = 0

  results = model.predict(
    source=str(dataset_images_dir),
    save=True,
    save_txt=True,
    conf=args.model_confidence,
    project=str(pseudo_labels_dir),
    imgsz=args.model_image_size
  )
    
  for r in results:
    detections += len(r.boxes)

  print(f"Total detections: {detections}")


def merge():
  if not os.path.exists(dataset_images_dir) or not os.path.exists(dataset_labels_dir):
    print(dataset_doesnt_exist_msg)
    return

  root_pseudo_labels_dir = dataset_dir / "pseudo_labels"
  pseudo_labels_dir = root_pseudo_labels_dir / "predict/labels"

  print(f"Merging pseudo-labels at {pseudo_labels_dir}...")
  for file in tqdm(os.listdir(pseudo_labels_dir)):
    dst_img = dataset_labels_dir / file
    if dst_img.exists():
      src_img_file = open(pseudo_labels_dir / file, "r")
      src_img_content = src_img_file.read()
      remapped_content = re.sub(r'^0\s', f'{args.id} ', src_img_content, flags=re.MULTILINE)

      with open(dst_img, "a") as dst_file:
        dst_file.write("\n" + remapped_content)
      src_img_file.close()

  shutil.rmtree(root_pseudo_labels_dir) # Cleanup pseudo-labels after merging

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