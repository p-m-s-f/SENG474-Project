# SENG474-Project

A wildfire image-classification project by Parris Mook-Sang-Forbes, Max Patchell, and Varun Menon.

| Table of Contents |
| ----------- |
| [Overview](#overview) |
| [Current Status](#current-status) |
| [Setup](#setup) |
| [Preparing the Images](#preparing-the-images) |
| [Dataset Splits](#dataset-splits) |
| [Obtaining Trained Checkpoints](#obtaining-trained-checkpoints) |
| [Quickstart](#quickstart) |
| [Reproduce Training](#reproduce-training) |
| [Predicting One Image](#predicting-one-image) |
| [Running the Visual Demo](#running-the-visual-demo) |
| [Evalulate the Cascade](#evaluate-the-cascade) |
| [Initial Validation Results](#initial-validation-results) |
| [Known Limitations](#known-limitations) |
| [File Overview](#file-overview) |
| [References](#references) |
| [AI Assistance](#ai-assistance) |

## Overview

The system uses two separately trained, pretrained ResNet-18 networks in a cascade:

1. A detector determines if an image contains **fire** or **no fire**.
2. Images determined to contain fire are passed to a risk classifier, which assigns an apparent visual risk of **low**, **medium**, or **high**.

The final outputs are `no_fire`, `fire_low`, `fire_medium`, and `fire_high`.

The dataset contains both aerial and ground-based images. Risk labels are manually assigned using visible smoke, flames, and foliage damage. Risk labels are visual assessments, not verified operational severity measurements.

## Current Status

Implemented:

- CSV label export and duplicate screening.
- Saved training, validation, and test splits.
- Dataset preparation from a downloaded Kaggle dataset.
- Stage-specific image loading and preprocessing.
- Training and best-checkpoint saving for both models.
- Single-image prediction.
- A visual demonstration using validation images.
- Full-cascade evaluation with saved metrics and predictions.
- A unified command interface through `src/main.py`.
- A tested quickstart that runs a sample prediction and full validation evaluation.

## Setup

Run all commands from the repository root (`SENG474-Project`), with your Python environment active.

For a first run:

1. Clone the repository and install dependencies using the steps below.
2. Download/extract the Kaggle dataset and run [dataset preparation](#prepare-the-images).
3. Download the trained models from the [checkpoint section](#obtain-trained-checkpoints).
4. Run the [quickstart](#quickstart).

The quickstart checks prerequisites and demonstrates the prepared system. It does not install packages, download data or weights, or train models automatically.

### 1. Clone the repository

```bash
git clone https://github.com/p-m-s-f/SENG474-Project.git
cd SENG474-Project
```

If already cloned, open your existing project folder and update it as appropriate.

### 2. Choose a Python environment

The initial setup was tested with **Python 3.14.7**, **PyTorch 2.14.1**, and **Torchvision 0.29.1**.

Use your preferred environment. If you do not have one:

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Check that your chosen Python version is compatible with the dependencies. The commands above use whichever Python installation they resolve to; they do not install Python 3.14.

### 3. Install dependencies

With your environment active:

```bash
python -m pip install -r requirements.txt
```

Package availability depends on Python version, operating system, and processor architecture. The following commands identify the active Python and pip versions when troubleshooting installation:

```bash
python --version
python -m pip --version
```

The scripts select NVIDIA CUDA when available, otherwise Apple MPS, otherwise CPU. CPU execution is supported but may be slower.

### 4. Check available commands

```bash
python src/main.py --help
```

For help with a particular command:

```bash
python src/main.py prepare --help
python src/main.py train --help
python src/main.py predict --help
python src/main.py demo --help
python src/main.py evaluate --help
```

## Preparing the Images

Download and extract the [Wildfire Dataset from Kaggle](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset).

The shared `image_labels.csv` contains filenames and manually assigned categories. The preparation script searches the extracted download and copies the required images into the matching project folders.

First check without copying:

```bash
python src/main.py prepare --source "/path/to/extracted/dataset" --dry-run
```

Then prepare the files:

```bash
python src/main.py prepare --source "/path/to/extracted/dataset"
```

Replace the example path with the actual extracted folder, not the ZIP. Keep the original download separate from the project's `data` folder.

The script preserves the source files, skips identical prepared files, and reports missing or ambiguous filenames. Copying requires additional disk space.

Expected local structure:

```text
data/
├── no_fire/
├── fire_low/
├── fire_medium/
└── fire_high/
```

Images remain local and are excluded from Git.

## Dataset Splits

The labelled dataset contains 1,040 images after removing four manually confirmed duplicate copies.

| Category | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| No fire | 423 | 90 | 92 | 605 |
| Low | 78 | 16 | 18 | 112 |
| Medium | 163 | 35 | 36 | 234 |
| High | 62 | 13 | 14 | 89 |
| **Total** | **726** | **154** | **160** | **1,040** |

Assignments are stored in `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`. These committed assignments define the documented experiment and are reused across both cascade stages.

The detector uses all examples in the selected split. The risk model uses only its fire examples.

## Obtaining Trained Checkpoints

Prediction, demonstration, and evaluation require:

```text
checkpoints/
├── detector_best.pt
└── risk_best.pt
```

**Google Drive download:** [Download the baseline checkpoint ZIP](https://drive.google.com/file/d/1zwEkgSP_wCIjfttBFOi2-4NnOsvt_1Id/view?usp=sharing)

Checkpoint files are excluded from Git. Download and extract the ZIP. The extracted folder is named `checkpoints` and contains `detector_best.pt` and `risk_best.pt`.

Place the entire extracted `checkpoints` folder in the project root, alongside `src`:

```text
SENG474-Project/
├── src/
├── checkpoints/
│   ├── detector_best.pt
│   └── risk_best.pt
└── README.md
```

If the project already has a `checkpoints` folder, copy the two `.pt` files from the extracted folder into the existing one instead. The files must be directly under `checkpoints/`, not nested under `checkpoints/checkpoints/`.

Checkpoints can also be generated using the [training commands](#reproduce-training). Prediction, demo, evaluation, and quickstart load existing checkpoints; training is a separate operation.

Checkpoints are tied to the model architecture, class ordering, and preprocessing used during training. The baseline checkpoints correspond to the configuration and validation results documented below. Source code and trained weights are distributed separately.

## Quickstart

After dependencies, images, and both checkpoints are prepared, run:

```bash
python src/quickstart.py
```

The script:

1. Checks that PyTorch and Torchvision load.
2. Checks that both checkpoint files and all validation image paths exist.
3. Runs the cascade on the first validation image.
4. Evaluates the full cascade on all 154 validation images.
5. Saves evaluation outputs to `outputs/val/`.

A successful run ends with:

```text
Quickstart completed successfully.
Prediction and validation evaluation both ran successfully.
```

This demonstrates functionality. The sample prediction does not have to match its reference label for the program to run successfully. The full validation metrics describe model performance.

The quickstart does **not** retrain either model or evaluate the test split. It overwrites existing validation result files. If prerequisites are missing, follow the reported instructions and rerun it.

With the supplied baseline weights and unchanged data, the recorded run produced **83.77% validation accuracy** and **71.88% macro-F1**. Numerical results may vary across platforms; checkpoint and split identifiers are recorded in `metrics.json`.

## Reproduce Training

Train the detector:

```bash
python src/main.py train --stage detector
```

Train the risk classifier:

```bash
python src/main.py train --stage risk
```

Current settings:

- ImageNet-pretrained ResNet-18, with all layers fine-tuned.
- Input: RGB images resized to 224 × 224 with ImageNet normalization.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 16.
- Epochs: 5.
- Loss: cross-entropy.
- Random seed: 0.
- Checkpoint selection: lowest validation loss.

Each training run starts from ImageNet-pretrained weights and saves the selected stage to `checkpoints/<stage>_best.pt`, replacing an existing file at that path. Checkpoint selection uses the lowest validation loss within the current run; existing checkpoints are not used to resume training.

Seeds help repeatability, but results may differ across hardware and software versions.

## Predicting One Image

```bash
python src/main.py predict --image "/path/to/image.jpg"
```

The image can be outside the dataset and does not need a reference label.

Prediction loads the saved models without training them. Only image pixels are supplied to the networks; folder names do not determine predictions.

## Running the Visual Demo

```bash
python src/main.py demo --predicted-category fire_high --seed 0
```

Available categories:

- `no_fire`
- `fire_low`
- `fire_medium`
- `fire_high`

The demo scans validation images, selects an example predicted as the requested category, and opens it. The terminal prints both the assigned label and prediction.

The seed controls example selection. The demo displays an image alongside its prediction and reference label, allowing visual inspection of correct and incorrect classifications. It does not replace full-set evaluation.

## Evaluate the Cascade

During development:

```bash
python src/main.py evaluate --split val
```

After model configurations are finalized:

```bash
python src/main.py evaluate --split test
```

Validation is used during development. The test split is reserved for assessment after model configuration has been finalized.

Evaluation includes all images in the selected split, including fires incorrectly rejected by the detector. It reports accuracy, per-class precision/recall/F1, macro-F1, the confusion matrix, false alarms, and missed fires.

Results are saved under:

```text
outputs/<split>/
├── predictions.csv
├── confusion_matrix.csv
└── metrics.json
```

Metrics include checkpoint identification information. Evaluation overwrites existing output files for the selected split.

## Initial Validation Results

Selected detector checkpoint: epoch 2, validation accuracy 94.16%.

Selected risk checkpoint: epoch 3, validation accuracy 75.00% on known fire images.

Full-cascade validation results:

| Metric | Result |
|---|---:|
| Accuracy | 83.77% |
| Macro-F1 | 71.88% |
| False alarms | 4 of 90 no-fire images |
| Missed fires | 5 of 64 fire images |
| Fire detection recall | 92.19% |

These are initial validation results, not final test results.

## Known Limitations

- The risk model overfits: training accuracy reaches 100%, while validation performance declines after epoch 3. Selecting an earlier checkpoint does not solve the underlying issue.
- A fire incorrectly classified as no fire never reaches the risk model.
- A non-fire image incorrectly passed through receives a fire-risk category.
- Risk labels are subjective visual assessments.
- Categories are imbalanced; overall accuracy hides weaker categories. Low-category validation recall is 43.75%.
- Splitting is by image rather than incident, so related photographs may cross splits.
- Duplicate screening does not guarantee all related images were identified.
- Resizing can remove small fire details and distort image proportions.
- The prototype is not validated for operational firefighting decisions.

## File Overview

| File | Purpose |
|---|---|
| `src/main.py` | Unified command entry point |
| `src/quickstart.py` | Check prerequisites, run a sample prediction, and evaluate validation data |
| `src/prepare_data.py` | Organize downloaded images using shared CSV labels |
| `src/export_labels.py` | Export labels from locally sorted folders |
| `src/check_duplicates.py` | Screen exact and visually similar image pairs |
| `src/split_data.py` | Generate repeatable split assignments |
| `src/dataset.py` | Load and preprocess images with stage-specific targets |
| `src/model.py` | Build ResNet-18 with the required output count |
| `src/train.py` | Train one stage and save its best checkpoint |
| `src/predict.py` | Load checkpoints and predict one image through the cascade |
| `src/demo.py` | Display validation examples filtered by prediction |
| `src/evaluate.py` | Evaluate the full cascade and save results |

`export_labels.py` and `split_data.py` overwrite their CSV outputs. They are preparation utilities, not commands to rerun before every prediction or training run.

## References

- El-Madafri, I., Peña, M., and Olmedo-Torre, N. (2023). *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* https://doi.org/10.3390/f14091697
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition.* https://doi.org/10.1109/CVPR.2016.90
- [Torchvision ResNet-18 documentation and pretrained weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)

## AI Assistance

OpenAI Codex assisted with programming explanations, code development, debugging, and documentation formatting.
