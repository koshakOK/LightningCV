def dice_loss(pred, target, smooth=1.):
    """[summary]

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): ground thuth
        smooth ([type], optional):  defaults to 1..

    Returns:
        torch.Tensor: mean loss throw batch
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
            (pred.sum(dim=2).sum(dim=2)
            + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
