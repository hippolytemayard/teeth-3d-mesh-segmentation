import logging
from pathlib import Path
from omegaconf import DictConfig

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim

from teeth_3d_seg.metrics.metrics import weighting_DSC, weighting_PPV, weighting_SEN
from teeth_3d_seg.training.losses import Generalized_Dice_Loss
from teeth_3d_seg.training.validation import validate


def train(
    model: nn.Module,
    optimizer: optim,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: DictConfig,
    device,
    writer=None,
) -> None:
    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

    best_val_dsc = 0.0

    class_weights = torch.ones(17).to(device, dtype=torch.float)

    logging.info("Starting training")

    for epoch in tqdm(range(config.train.epochs)):
        model.train()

        loss_epoch, mdsc_epoch, msen_epoch, mppv_epoch = training_loop(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            config=config,
            class_weights=class_weights,
            epoch=epoch,
            writer=writer,
        )

        losses.append(loss_epoch)
        mdsc.append(mdsc_epoch)
        msen.append(msen_epoch)
        mppv.append(mppv_epoch)

        val_loss_epoch, val_mdsc_epoch, val_msen_epoch, val_mppv_epoch = validate(
            model=model,
            val_loader=validation_loader,
            config=config,
            device=device,
            class_weights=class_weights,
            epoch=epoch,
        )

        val_losses.append(val_loss_epoch)
        val_mdsc.append(val_mdsc_epoch)
        val_msen.append(val_msen_epoch)
        val_mppv.append(val_mppv_epoch)

        if config.train.debug:
            print(
                "*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****".format(
                    epoch + 1,
                    config.train.epochs,
                    losses[-1],
                    mdsc[-1],
                    msen[-1],
                    mppv[-1],
                    val_losses[-1],
                    val_mdsc[-1],
                    val_msen[-1],
                    val_mppv[-1],
                )
            )

        if writer is not None:
            writer.add_scalar("loss train", losses[-1], epoch + 1)
            writer.add_scalar("DSC train", mdsc[-1], epoch + 1)
            writer.add_scalar("SEN train", msen[-1], epoch + 1)
            writer.add_scalar("PPV train", mppv[-1], epoch + 1)
            writer.add_scalar("loss validation", val_losses[-1], epoch + 1)
            writer.add_scalar("DSC validation", val_mdsc[-1], epoch + 1)
            writer.add_scalar("SEN validation", val_msen[-1], epoch + 1)
            writer.add_scalar("PPV validation", val_mppv[-1], epoch + 1)

        # save the checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "mdsc": mdsc,
                "msen": msen,
                "mppv": mppv,
                "val_losses": val_losses,
                "val_mdsc": val_mdsc,
                "val_msen": val_msen,
                "val_mppv": val_mppv,
            },
            config.save_dir + f"checkpoint_epoch{epoch}.pt",
        )

        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "mdsc": mdsc,
                    "msen": msen,
                    "mppv": mppv,
                    "val_losses": val_losses,
                    "val_mdsc": val_mdsc,
                    "val_msen": val_msen,
                    "val_mppv": val_mppv,
                },
                config.save_dir + "{}_best.pt".format("best_model"),
            )

        # save all losses and metrics data
        pd_dict = {
            "loss": losses,
            "DSC": mdsc,
            "SEN": msen,
            "PPV": mppv,
            "val_loss": val_losses,
            "val_DSC": val_mdsc,
            "val_SEN": val_msen,
            "val_PPV": val_mppv,
        }
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(f"losses_metrics_vs_epoch_{epoch}.csv")


def training_loop(loader, model, optimizer, device, config, class_weights, epoch, writer=None):
    # training
    model.train()

    running_loss = 0.0
    running_mdsc = 0.0
    running_msen = 0.0
    running_mppv = 0.0

    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0

    for i_batch, batched_sample in enumerate(loader):
        # send mini-batch to device
        inputs = batched_sample["cells"].to(device, dtype=torch.float)
        labels = batched_sample["labels"].to(device, dtype=torch.long)
        A_S = batched_sample["A_S"].to(device, dtype=torch.float)
        A_L = batched_sample["A_L"].to(device, dtype=torch.float)
        one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=config.model.num_classes)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, A_S, A_L)
        loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_mdsc += dsc.item()
        running_msen += sen.item()
        running_mppv += ppv.item()

        loss_epoch += loss.item()
        mdsc_epoch += dsc.item()
        msen_epoch += sen.item()
        mppv_epoch += ppv.item()

        if i_batch % config.train.num_batches_to_print == config.train.num_batches_to_print - 1:
            if config.train.debug:
                print(
                    "[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}".format(
                        epoch + 1,
                        config.train.epochs,
                        i_batch + 1,
                        len(loader),
                        running_loss / config.train.num_batches_to_print,
                        running_mdsc / config.train.num_batches_to_print,
                        running_msen / config.train.num_batches_to_print,
                        running_mppv / config.train.num_batches_to_print,
                    )
                )

            if writer is not None:
                writer.add_scalar(
                    "loss_batch",
                    running_loss / config.train.num_batches_to_print,
                    int(epoch + (i_batch + 1) / len(loader)),
                )
                writer.add_scalar(
                    "DSC_batch",
                    running_mdsc / config.train.num_batches_to_print,
                    int(epoch + (i_batch + 1) / len(loader)),
                )
                writer.add_scalar(
                    "SEN_batch",
                    running_msen / config.train.num_batches_to_print,
                    int(epoch + (i_batch + 1) / len(loader)),
                )
                writer.add_scalar(
                    "PPV_batch",
                    running_mppv / config.train.num_batches_to_print,
                    int(epoch + (i_batch + 1) / len(loader)),
                )

            running_loss = 0.0
            running_mdsc = 0.0
            running_msen = 0.0
            running_mppv = 0.0

    return (
        loss_epoch / len(loader),
        mdsc_epoch / len(loader),
        msen_epoch / len(loader),
        mppv_epoch / len(loader),
    )
