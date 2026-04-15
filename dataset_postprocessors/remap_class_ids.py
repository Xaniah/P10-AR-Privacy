from pathlib import Path
from functools import partial
from multiprocessing import Pool
import re

from tqdm import tqdm

def _process_file(label_file: Path, class_id_mapping: dict[int, int]) -> None:
  pattern = re.compile(r'^\d+', re.MULTILINE)

  def replace(match):
      old_id = int(match.group())
      if old_id in class_id_mapping:
          return str(class_id_mapping[old_id])
      print(f"Warning: Class ID {old_id} in file {label_file} not found in mapping, keeping original.")
      return match.group()

  content = label_file.read_text()
  new_content = pattern.sub(replace, content)
  if content != new_content:
      label_file.write_text(new_content)


def remap_class_ids(dataset_dir_labels: Path, class_id_mapping: dict[int, int]) -> None:
  """
  Remaps class IDs in YOLO format annotation files based on the provided mapping.
  Args:
      dataset_dir_labels: Path to the directory containing the YOLO annotation files
      class_id_mapping: A dictionary mapping old class IDs to new class IDs
  """
  print("Remapping class IDs in annotation files...")
  label_files = list(dataset_dir_labels.glob("*.txt"))

  process_file = partial(_process_file, class_id_mapping=class_id_mapping)
  with Pool() as pool:
      list(tqdm(pool.imap(process_file, label_files), total=len(label_files)))

  print("Class ID remapping completed.")