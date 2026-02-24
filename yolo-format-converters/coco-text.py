import os

from pylabel import importer
from ultralytics.utils.downloads import download
import shutil
from pathlib import Path

dir = Path(os.path.dirname(os.path.realpath(__file__)))  # dataset root dir

train_images = "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip"
target_dir = dir.parent / "datasets/coco-text"
download_dir = dir.parent / "datasets/coco-text-download"

download(train_images, dir=download_dir, threads=1, delete=True)

importer.ImportCoco(download_dir / "cocotext.v2.json").export.ExportToYoloV5(
    output_path=target_dir,
    copy_images=True,
    use_splits=True,
    cat_id_index=0,
)

if os.path.exists(download_dir):
    shutil.rmtree(download_dir)