# OCT Retinal Layer Segmentation

Deep learning-based segmentation of retinal layers in Spectral-Domain Optical Coherence Tomography (SD-OCT) B-scans, built with PyTorch.

## Overview

Optical Coherence Tomography (OCT) is the standard imaging modality for diagnosing and monitoring retinal disease, and automated segmentation of individual retinal layers is a key building block for quantifying layer thickness and detecting pathology such as diabetic macular edema (DME). This project trains a U-Net to segment OCT B-scans into retinal layer regions, using the Duke SD-OCT dataset for DME as a testbed.

## Dataset

- **Source**: Duke SD-OCT dataset for Diabetic Macular Edema (Chiu et al., *Biomedical Optics Express*, 2015) — `datasets/duke/2015_BOE_Chiu/`.
- **Subjects**: 10 subjects, each with 61 B-scans (496 × 768, grayscale) and two independent manual boundary annotations (grader 1, grader 2).
- **Annotated scans only**: most B-scans per subject are unannotated; only ~11 scans per subject have manual boundary labels, giving ~110 usable annotated scans (grader 1) across all subjects.
- **Masks**: the 8 manually-annotated boundaries are converted into dense pixel-wise masks (`src/masks.py`) — background (class 0) plus 7 inter-boundary retinal layer regions (classes 1–7).
- **Split**: subject-level 80/20 train/validation split (`src/training/split.py`, `random_state=42`) — splitting by subject rather than by scan avoids leaking B-scans from the same eye across train and validation.

## Method

### Architecture

A 4-level U-Net (`src/models/unet.py`, built from `DoubleConv`/`DownBlock`/`UpBlock` modules):

- Input: single-channel 496×768 OCT B-scan.
- Encoder: 1→32→64→128→256→512 channels, each stage a `Conv-BN-ReLU ×2` block followed by max-pooling.
- Decoder: transposed-convolution upsampling with skip connections back to the encoder, mirroring the encoder channel widths.
- Output: 1×1 convolution to 8-class logits (background + 7 retinal layer regions), same spatial resolution as the input.

### Loss

Training started with plain `CrossEntropyLoss`. A combined Dice + Cross-Entropy loss (`DiceCrossEntropyLoss` in `src/training/losses.py`) was implemented as a follow-up experiment — softmax Dice over one-hot targets, averaged 50/50 with cross-entropy. In practice this combined loss **did not outperform** the cross-entropy-trained baseline on this dataset; the checkpoint from that run (`checkpoints/dice_ce_model.pth`) is kept for reference, but the reported results below are from the baseline model (`checkpoints/best_model.pth`).

### Training

- Optimizer: Adam, learning rate `1e-3`.
- Epochs: 20, batch size: 4.
- No data augmentation.
- Device: auto-selects Apple MPS → CUDA → CPU.

## Results

Evaluated on 22 held-out validation B-scans (2 held-out subjects):

| Metric | Value |
|---|---|
| Mean Dice | 0.8282 |
| Mean IoU | 0.7188 |
| Pixel Accuracy | 0.9720 |

Per-class Dice (class 0 = background, classes 1–7 = retinal layer regions):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Dice | 0.9899 | 0.8001 | 0.8870 | 0.7587 | 0.7282 | 0.8603 | 0.8299 | 0.8085 |

Layer 4 is the weakest-performing region, while the background class (unsurprisingly) is segmented almost perfectly.

Qualitative best/average/worst-case comparisons (original scan / ground truth / prediction / overlay):

![Representative segmentation results](figures/representative_segmentation_results.png)

## Repository structure

```
src/
├── io.py                 # .mat subject loading
├── masks.py               # boundary annotations -> pixel-wise masks
├── dataset.py              # OCTLayerDataset (PyTorch Dataset)
├── visualization.py
├── models/                 # UNet, DoubleConv, DownBlock, UpBlock
└── training/
    ├── train.py             # training entrypoint
    ├── trainer.py            # train_one_epoch
    ├── evaluate.py            # evaluate
    ├── losses.py               # CrossEntropy / DiceCrossEntropyLoss
    ├── metrics.py               # pixel_accuracy
    ├── loaders.py                # DataLoader construction
    └── split.py                   # subject-level train/val split

notebooks/
├── 01_dataset_exploration.ipynb
├── 02_generate_masks.ipynb
├── 03_preprocessing.ipynb
├── 04_dataloader.ipynb
├── 05_unet_forward_pass.ipynb
├── 06_loss_function.ipynb
├── 07_inference.ipynb
└── 08_evaluation.ipynb
```

## Setup

Dependencies actually used by this codebase: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`. There is no `requirements.txt` yet — install these with `pip` in a fresh environment before running the notebooks or `src/training/train.py`.

The Duke dataset (`.mat` files) is not included in this repository; place it under `datasets/duke/2015_BOE_Chiu/` before running the dataset/training notebooks.

## Limitations & future work

- Macro-averaged Dice/IoU are pulled up by the easy, dominant background class — per-class numbers are more informative than the mean.
- No data augmentation is applied during training.
- The combined Dice+CE loss did not improve over cross-entropy alone in this setup and was not adopted.
- Only grader-1 annotations are used; inter-grader variability (grader 2) is not evaluated.
- Segmentation classes are positional (layer 0–7) rather than mapped to anatomical layer names (e.g. ILM, RNFL, RPE).

## License

Code is released under the MIT License (see `LICENSE`). The Duke SD-OCT dataset is a separate third-party resource — cite Chiu et al., *Biomedical Optics Express* 2015 if you use it in research.
