import torch

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_CLASSES = 1
MAX_EPOCHS = 50
SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Torch realization of iou metric.
        params:
            outputs shape: BATCH x H x W
            labels shape: BATCH x H x W
        return: IoU score
    """

    intersection = (outputs.long() & labels.long()).float().sum((1, 2))
    union = (outputs.long() | labels.long()).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded.mean()
