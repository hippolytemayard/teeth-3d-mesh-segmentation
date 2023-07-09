from pathlib import Path

import torch
from torch.utils.data import Dataset
from vedo import load
import json


class Mesh_Dataset(Dataset):
    def __init__(self, data_folder: str, num_classes: int = 15, patch_size: int = 7000):
        self.data_list = list(Path(data_folder).glob("**/*.obj"))
        self.label_list = list(Path(data_folder).glob("**/*.json"))

        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mesh = load(self.data_list[idx])
        labels = json.load(labels)
