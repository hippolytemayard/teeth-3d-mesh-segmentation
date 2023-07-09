import torch
from teeth_3d_seg.training.losses import Generalized_Dice_Loss
from teeth_3d_seg.metrics.metrics import weighting_DSC, weighting_PPV, weighting_SEN
from torch import nn


def validate(model, val_loader, config, device, class_weights, epoch):
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_val_mdsc = 0.0
        running_val_msen = 0.0
        running_val_mppv = 0.0
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0
        for i_batch, batched_val_sample in enumerate(val_loader):
            inputs = batched_val_sample["cells"].to(device, dtype=torch.float)
            labels = batched_val_sample["labels"].to(device, dtype=torch.long)
            A_S = batched_val_sample["A_S"].to(device, dtype=torch.float)
            A_L = batched_val_sample["A_L"].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=config.model.num_classes)

            outputs = model(inputs, A_S, A_L)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

            running_val_loss += loss.item()
            running_val_mdsc += dsc.item()
            running_val_msen += sen.item()
            running_val_mppv += ppv.item()
            val_loss_epoch += loss.item()
            val_mdsc_epoch += dsc.item()
            val_msen_epoch += sen.item()
            val_mppv_epoch += ppv.item()

            if (
                i_batch % config.train.num_batches_to_print == config.train.num_batches_to_print - 1
            ):  # print every N mini-batches
                print(
                    "[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}".format(
                        epoch + 1,
                        config.train.epochs,
                        i_batch + 1,
                        len(val_loader),
                        running_val_loss / config.train.num_batches_to_print,
                        running_val_mdsc / config.train.num_batches_to_print,
                        running_val_msen / config.train.num_batches_to_print,
                        running_val_mppv / config.train.num_batches_to_print,
                    )
                )
                running_val_loss = 0.0
                running_val_mdsc = 0.0
                running_val_msen = 0.0
                running_val_mppv = 0.0

    return (
        val_loss_epoch / len(val_loader),
        val_mdsc_epoch / len(val_loader),
        val_msen_epoch / len(val_loader),
        val_mppv_epoch / len(val_loader),
    )
