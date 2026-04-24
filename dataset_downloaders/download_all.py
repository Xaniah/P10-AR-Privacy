import sys
import os

# Add project root to path so package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_downloaders.gtsdb import download_gtsdb
from dataset_downloaders.human_faces_1_roboflow import download_human_faces_1_roboflow
from dataset_downloaders.human_faces_2_roboflow import download_human_faces_2_roboflow
from dataset_downloaders.license_plate_roboflow import download_license_plate_roboflow
from dataset_downloaders.license_plate_UC3M_LP import download_license_plate_uc3m_lp
from dataset_downloaders.wider_face import download_wider_face
#from dataset_downloaders.coco_text import download_coco_text
from dataset_downloaders.open_images_v7 import download_open_images_v7
from utils import split_datasets
import argparse

parser = argparse.ArgumentParser(description="Download all datasets")
parser.add_argument("--no-split", action="store_true", help="Whether to skip splitting the datasets after downloading")

def download_all_datasets(split_datasets_after_download=True):
  download_wider_face()
  # download_coco_text()
  download_open_images_v7()
  download_human_faces_1_roboflow()
  download_human_faces_2_roboflow()
  download_license_plate_roboflow()
  download_license_plate_uc3m_lp()
  download_gtsdb()

  if split_datasets_after_download:
    split_datasets()

if __name__ == "__main__":
  args = parser.parse_args()
  download_all_datasets(split_datasets_after_download=not args.no_split)