import sys
import os

# Add project root to path so package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_downloaders.wider_face import download_wider_face
from dataset_downloaders.coco_text import download_coco_text
from dataset_downloaders.open_images_v7 import download_open_images_v7
from utils import split_datasets

def download_all_datasets():
    download_wider_face()
    # download_coco_text()
    download_open_images_v7()
    split_datasets()

if __name__ == "__main__":
    download_all_datasets()