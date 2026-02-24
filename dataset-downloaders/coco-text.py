import os
from pathlib import Path
import kagglehub

dir = Path(os.path.dirname(os.path.realpath(__file__)))  # dataset root dir
target_dir = dir.parent / "datasets/coco-text"

# Download latest version
path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3", output_dir=target_dir)

print("Path to dataset files:", path)