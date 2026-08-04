import torch


def pixel_accuracy(predictions, targets):

    predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum()

    total = targets.numel()

    return correct.float() / total