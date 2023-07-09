import logging
from pathlib import Path

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from teeth_3d_seg.configs.settings import CONFIG_PATH, DATA_PATH
from teeth_3d_seg.dataset.mesh_loader import get_mesh_loader
from teeth_3d_seg.model.meshsegnet import MeshSegNet
from teeth_3d_seg.training.training import train
from teeth_3d_seg.utils.file import load_yaml

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_yaml(path=CONFIG_PATH)

    # TODO include validation to loader
    val_dir = "/data/ubuntu/data/Teeth3DS/test_set/"
    save_dir = Path(config.save_dir)
    experiment_dir = save_dir / f"experiment{config.experiment}"
    save_dir.mkdir(exist_ok=True)
    experiment_dir.mkdir(exist_ok=True)

    logging.info("Training loader initialization")

    train_loader = get_mesh_loader(
        data_folder=DATA_PATH,
        label_encoder=config.data.label_encoder,
        batch_size=config.data.loader.train.batch_size,
        shuffle=config.data.loader.train.shuffle,
        patch_size=config.data.dataset.patch_size,
        num_workers=config.data.loader.num_workers,
    )

    logging.info("Validation loader initialization")
    validation_loader = get_mesh_loader(
        data_folder=val_dir,
        label_encoder=config.data.label_encoder,
        batch_size=config.data.loader.validation.batch_size,
        shuffle=config.data.loader.validation.shuffle,
        patch_size=config.data.dataset.patch_size,
        num_workers=config.data.loader.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MeshSegNet(
        num_classes=config.model.num_classes,
        num_channels=config.model.num_channels,
        with_dropout=config.model.dropout,
        dropout_p=config.model.dropout_proba,
    ).to(device, dtype=torch.float)

    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr, amsgrad=True)

    writer = SummaryWriter(log_dir=experiment_dir) if config.train.tensorboard else None

    logging.info("Starting training")

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
        writer=writer,
    )
