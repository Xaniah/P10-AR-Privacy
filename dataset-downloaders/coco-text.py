import os
from pathlib import Path
import json
import kagglehub
from ultralytics.utils.downloads import download
import shutil
from tqdm import tqdm
from collections import defaultdict

dir = Path(os.path.dirname(os.path.realpath(__file__)))  # dataset root dir
target_dir = dir.parent / "datasets/coco-text"
labels = "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip"
labels_path = target_dir / "cocotext.v2.json"

class_mappings = {
    "machine printed": 0,
    "handwritten": 1,
    "others": 2,
}

if os.path.exists(target_dir / "labels"):
    shutil.rmtree(target_dir / "labels")

def download_images():
    # Download latest version
    images_path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3", output_dir=target_dir)

    if os.path.exists(target_dir / "coco2014/images"):
        os.rename(target_dir / "coco2014/images", target_dir / "images")

    if os.path.exists(target_dir / "images/test2014"):
        os.remove(target_dir / "images/test2014")

    if os.path.exists(target_dir / "coco2014"):
        shutil.rmtree(target_dir / "coco2014")



def download_labels():
    os.makedirs(target_dir / "labels/train2014", exist_ok=True)
    os.makedirs(target_dir / "labels/val2014", exist_ok=True)

    if not os.path.exists(labels_path):
        download(labels, dir=target_dir, threads=1, delete=True)

    with open(labels_path, "r") as f:
        content = json.load(f)

    annotations = content["anns"]
    images = content["imgs"]
    strs_to_write = defaultdict(list)

    for annotation in annotations.values():
        image_id = annotation["image_id"]
        x, y, w, h = annotation["bbox"]
        image = images[str(image_id)]
        image_set = image["set"]

        image_width = image["width"]
        image_height = image["height"]

        strs_to_write[image_id].append(f"{class_mappings[annotation["class"]]} {(x + w / 2.0) / image_width} {(y + h / 2.0) / image_height} {w / image_width} {h / image_height}")


    for image_id, str_to_write in tqdm(strs_to_write.items()):
        image = images[str(image_id)]
        image_set = image["set"]
        with open(f"{target_dir}/labels/{image_set}2014/{os.path.splitext(image["file_name"].replace("train", image_set))[0] + ".txt"}", "a+") as f:
            f.write("\n".join(str_to_write))

    os.remove(labels_path)

download_images()
download_labels()