# Source: https://universe.roboflow.com/whtmidoin/face-detection-fyp

import os
from pathlib import Path

from roboflow import Roboflow
from dotenv import load_dotenv
from ultralytics import SETTINGS

from dataset_postprocessors import remap_class_ids
from utils import get_dataset_class_id_by_name

def download_human_faces_2_roboflow():
  load_dotenv()

  ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

  if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in the .env file.")

  dataset_path = Path(SETTINGS["datasets_dir"]) / "Human-Faces-1"
  
  if not dataset_path.exists():    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("whtmidoin").project("face-detection-fyp")
    version = project.version(1)
    version.download("yolo26", location=str(dataset_path), overwrite=False)

    # Remove the data.yaml file that Roboflow creates as it is not needed, because we have our own dataset-config.yaml file.
    data_yaml = dataset_path / "data.yaml"
    if (data_yaml).exists():
      os.remove(data_yaml)

    for split in "test", "train", "valid":
      remap_class_ids(dataset_path / split / "labels", {0: get_dataset_class_id_by_name("Face")})
  else:
    print(f"Roboflow human face 2 Dataset already exists at {dataset_path}, skipping download.")

if __name__ == "__main__":
  download_human_faces_2_roboflow()