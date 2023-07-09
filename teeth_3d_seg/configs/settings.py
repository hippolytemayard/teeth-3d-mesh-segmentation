from pathlib import Path

# Paths
ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = Path("/data/ubuntu/data/Teeth3DS/data_part_1/")  # ROOT_PATH / "data"
CONFIG_PATH = ROOT_PATH / "teeth_3d_seg" / "configs" / "training_config.yaml"
