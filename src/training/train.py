import torch

from src.models.unet import UNet
from src.io import load_dataset
from pathlib import Path
from src.training.split import split_subjects
from src.training.loaders import create_loaders
from src.training.losses import create_cross_entropy_loss
from src.training.trainer import train_one_epoch
from src.training.evaluate import evaluate


def main():

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")


    # -------------------------
    # Load dataset
    # -------------------------

    subjects = load_dataset(
        "datasets/duke/2015_BOE_Chiu"
    )

    print(
        f"Loaded {len(subjects)} subjects"
    )


    # -------------------------
    # Train/validation split
    # -------------------------

    train_subjects, val_subjects = split_subjects(
        subjects,
        test_size=0.2,
        random_state=42,
    )

    print(
        f"Train subjects: {len(train_subjects)}"
    )

    print(
        f"Validation subjects: {len(val_subjects)}"
    )


    # -------------------------
    # Data loaders
    # -------------------------

    train_loader, val_loader = create_loaders(
        train_subjects,
        val_subjects,
        batch_size=4,
        num_workers=0,
    )

    print(
        f"Training scans: {len(train_loader.dataset)}"
    )

    print(
        f"Validation scans: {len(val_loader.dataset)}"
    )


    # -------------------------
    # Model
    # -------------------------

    model = UNet(
        n_channels=1,
        n_classes=8,
    )

    model = model.to(device)


    # -------------------------
    # Loss
    # -------------------------

    criterion = create_cross_entropy_loss()


    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )


    # -------------------------
    # Training
    # -------------------------

    epochs = 20
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )


        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )


        print(
            f"""
Epoch {epoch+1}/{epochs}

Train Loss: {train_loss:.4f}
Val Loss:   {val_loss:.4f}
"""
        )


    print("Training complete")
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            save_dir / "best_model.pth",
        )

        print("✓ Saved new best model")


if __name__ == "__main__":
    main()