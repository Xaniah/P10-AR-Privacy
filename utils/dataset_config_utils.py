import yaml

def load_names(yaml_path: str) -> dict[int, str]:
  with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
  return data["names"]

yaml_names = load_names("dataset-config.yaml")
name_to_id = {v: k for k, v in yaml_names.items()}

def get_dataset_class_id_by_name(name: str) -> int:
  return name_to_id[name]
