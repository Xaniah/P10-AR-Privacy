# Source: https://medium.com/@ahmetcan7754/i-converted-the-wider-face-detection-dataset-to-yolo12-format-8b47d1ec8c88
import os
from PIL import Image
from pathlib import Path
from ultralytics.utils.downloads import download
import requests
import zipfile
import shutil

dir = Path(os.path.dirname(os.path.realpath(__file__)))  # dataset root dir

# Paths
train_images = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip"
val_images = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"

images_base_dir = [train_images, val_images]
target_dir = dir.parent / "datasets/WIDER-FACE"

source_labels = "https://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
labels_file_name_zip = f"{target_dir}/wider_face_split.zip"
labels_folder_name = os.path.splitext(labels_file_name_zip)[0]

annotations_to_img_path = {
    "wider_face_train_bbx_gt.txt": f"{target_dir}/WIDER_train",
    "wider_face_val_bbx_gt.txt": f"{target_dir}/WIDER_val",
}

def main():
    images_download_dir = []
    for base_dir in images_base_dir:
        file_name = base_dir.split("/")[-1]
        folder_name = os.path.splitext(file_name)[0]

        if not os.path.exists(target_dir / folder_name):
            images_download_dir.append(base_dir)

    download(images_download_dir, dir=target_dir, threads=3, delete=True)

    for base_dir in images_download_dir:
        file_name = base_dir.split("/")[-1]
        folder_name = os.path.splitext(file_name)[0]

        for root, _, files in os.walk(target_dir / folder_name / "images"):
            if not files:
                continue
            for file in files:
                root_path = Path(root)
                os.rename(root_path / file, root_path.parent / file)

            os.rmdir(root)    

    response = requests.get(source_labels, verify=False)
    with open(labels_file_name_zip, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(labels_file_name_zip, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    if os.path.exists(labels_file_name_zip):
        os.remove(labels_file_name_zip)


    for file_name in [file_name for file_name in os.listdir(labels_folder_name) if "_bbx_" in file_name]:
        if file_name not in annotations_to_img_path:
            print(f"Unable to find image path for {file_name}")
            continue

        # Read label file
        with open(labels_folder_name + f"/{file_name}", "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "/" in line or line.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.basename(line)
                i += 1
            else:
                i += 1
                continue

            # Get bounding boxes count
            bbox_count = int(lines[i].strip()) if lines[i].strip().isdigit() else 0
            i += 1

            bboxes = []
            for _ in range(bbox_count):
                bbox = list(map(float, lines[i].split()[:4]))
                bboxes.append(bbox)
                i += 1

            # Get image dimensions
            image_full_path = os.path.join(annotations_to_img_path[file_name] + "/images", image_path)
            with Image.open(image_full_path) as img:
                img_width, img_height = img.size


            # Convert to YOLO format
            yolo_lines = [f"0 {(x + w / 2.0) / img_width} {(y + h / 2.0) / img_height} {w / img_width} {h / img_height}" 
                        for x, y, w, h in bboxes]

            # Write to label file
            labels_path = annotations_to_img_path[file_name] + "/labels"
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            output_file = os.path.join(labels_path, os.path.splitext(image_path)[0] + ".txt")
            with open(output_file, "w") as out_f:
                out_f.write("\n".join(yolo_lines))

    if os.path.exists(labels_folder_name):
        shutil.rmtree(labels_folder_name)

if __name__ == "__main__":
    main()