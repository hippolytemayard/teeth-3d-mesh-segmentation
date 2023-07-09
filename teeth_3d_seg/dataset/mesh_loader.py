from torch.utils.data import DataLoader
from teeth_3d_seg.dataset.mesh_dataset import MeshDataset


def get_mesh_loader(
    data_folder: str, label_encoder: str, batch_size: int, shuffle: bool, patch_size: int, num_workers: int = 4
) -> DataLoader:
    """Initializing dataloader from data directory"""
    dataset = MeshDataset(data_folder=data_folder, patch_size=patch_size, label_encoder=label_encoder)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
