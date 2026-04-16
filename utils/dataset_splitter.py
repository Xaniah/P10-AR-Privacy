import os
from pathlib import Path
import shutil
from ultralytics.utils import SETTINGS
import random
from tqdm import tqdm

def split_dataset(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, subset_dir, max_samples=5000, train_split=0.9, val_split=0.1):
    train_image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.webp"))
    train_image_files = random.sample(train_image_files, min(len(train_image_files), round(max_samples*train_split)))

    val_image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.webp"))
    val_image_files = random.sample(val_image_files, min(len(val_image_files), round(max_samples*val_split)))

    for split, files, labels_dir in [("train", train_image_files, train_labels_dir), ("val", val_image_files, val_labels_dir)]:
        split_images_dir = subset_dir / "images" / split
        split_labels_dir = subset_dir / "labels" / split 
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

        print(f"Copying {len(files)} {split} images and labels from {labels_dir} to {split_labels_dir}...")
        for img_file in tqdm(files):
            shutil.copy(img_file, split_images_dir)
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy(label_file, split_labels_dir)


def split_datasets():
    MAX_SAMPLES = 15000000
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 1 - TRAIN_SPLIT

    DATASETS_DIR = Path(SETTINGS["datasets_dir"])

    WIDER_FACE_DIR = DATASETS_DIR / "WIDER-FACE"
    COCO_TEXT_DIR = DATASETS_DIR / "coco-text"
    OPEN_IMAGES_DIR = DATASETS_DIR / "open-images-v7"
    ROBOFLOW_LICENSE_PLATE_DIR = DATASETS_DIR / "License-Plate-Recognition-13"
    UC3M_LP_DIR = DATASETS_DIR / "UC3M-LP"
    GTSDB_DIR = DATASETS_DIR / "GTSDB"
    SUBSET_DIR = DATASETS_DIR / "subset"

    if not os.path.exists(WIDER_FACE_DIR):
        raise FileNotFoundError(f"WIDER FACE dataset directory '{WIDER_FACE_DIR}' does not exist.")
    # if not os.path.exists(COCO_TEXT_DIR):
    #     raise FileNotFoundError(f"COCO TEXT dataset directory '{COCO_TEXT_DIR}' does not exist.")
    if not os.path.exists(OPEN_IMAGES_DIR):
        raise FileNotFoundError(f"OPEN IMAGES dataset directory '{OPEN_IMAGES_DIR}' does not exist.")
    if not os.path.exists(ROBOFLOW_LICENSE_PLATE_DIR):
        raise FileNotFoundError(f"ROBOFLOW LICENSE PLATE dataset directory '{ROBOFLOW_LICENSE_PLATE_DIR}' does not exist.")
    if not os.path.exists(UC3M_LP_DIR):
        raise FileNotFoundError(f"UC3M LP dataset directory '{UC3M_LP_DIR}' does not exist.")
    if not os.path.exists(GTSDB_DIR):
        raise FileNotFoundError(f"GTSDB dataset directory '{GTSDB_DIR}' does not exist.")


    if os.path.exists(SUBSET_DIR):
        shutil.rmtree(SUBSET_DIR)

    os.mkdir(SUBSET_DIR)

    print("Splitting datasets...")    

    split_dataset(
        train_images_dir=WIDER_FACE_DIR / "WIDER_train" / "images",
        train_labels_dir=WIDER_FACE_DIR / "WIDER_train" / "labels",
        val_images_dir=WIDER_FACE_DIR / "WIDER_val" / "images",
        val_labels_dir=WIDER_FACE_DIR / "WIDER_val" / "labels",
        subset_dir=SUBSET_DIR,
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    # split_dataset(
    #     train_images_dir=COCO_TEXT_DIR / "images" / "train2014",
    #     train_labels_dir=COCO_TEXT_DIR / "labels" / "train2014",
    #     val_images_dir=COCO_TEXT_DIR / "images" / "val2014",
    #     val_labels_dir=COCO_TEXT_DIR / "labels" / "val2014",
    #     subset_dir=SUBSET_DIR,
    #     max_samples=MAX_SAMPLES,
    #     train_split=TRAIN_SPLIT,
    #     val_split=VAL_SPLIT,
    # )

    split_dataset(
        train_images_dir=OPEN_IMAGES_DIR / "images" / "train",
        train_labels_dir=OPEN_IMAGES_DIR / "labels" / "train",
        val_images_dir=OPEN_IMAGES_DIR / "images" / "val",
        val_labels_dir=OPEN_IMAGES_DIR / "labels" / "val",
        subset_dir=SUBSET_DIR,
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    split_dataset(
        train_images_dir=ROBOFLOW_LICENSE_PLATE_DIR / "train" / "images",
        train_labels_dir=ROBOFLOW_LICENSE_PLATE_DIR / "train" / "labels",
        val_images_dir=ROBOFLOW_LICENSE_PLATE_DIR / "valid" / "images",
        val_labels_dir=ROBOFLOW_LICENSE_PLATE_DIR / "valid" / "labels",
        subset_dir=SUBSET_DIR,
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    split_dataset(
        train_images_dir=UC3M_LP_DIR / "images" / "train",
        train_labels_dir=UC3M_LP_DIR / "labels" / "train",
        val_images_dir=UC3M_LP_DIR / "images" / "val",
        val_labels_dir=UC3M_LP_DIR / "labels" / "val",
        subset_dir=SUBSET_DIR,
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    split_dataset(
        train_images_dir=GTSDB_DIR / "GTSDB_Train_and_Test" / "Train" / "images",
        train_labels_dir=GTSDB_DIR / "GTSDB_Train_and_Test" / "Train" / "labels",
        val_images_dir=GTSDB_DIR / "GTSDB_Train_and_Test" / "Test" / "images",
        val_labels_dir=GTSDB_DIR / "GTSDB_Train_and_Test" / "Test" / "labels",
        subset_dir=SUBSET_DIR,
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

if __name__ == "__main__":
    split_datasets()