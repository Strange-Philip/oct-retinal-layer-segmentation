from torch.utils.data import DataLoader

from src.dataset import OCTLayerDataset


def create_loaders(
    train_subjects,
    val_subjects,
    batch_size=4,
    num_workers=0,
):
    train_dataset = OCTLayerDataset(
        train_subjects,
        grader=1,
    )

    val_dataset = OCTLayerDataset(
        val_subjects,
        grader=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
    )