def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth=1.0):
    """
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    """
    smooth = 1.0
    loss = 0.0
    n_classes = y_pred.shape[-1]

    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()

        # with weight
        w = class_weights[c] / class_weights.sum()
        loss += w * (1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))

    return loss
