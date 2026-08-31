# Ctrl-Alt-Elite — AI Image Detection

> A deep-learning classifier for distinguishing **AI-generated (`FAKE`)** images from **real (`REAL`)** images, with clean-set and corruption-robustness evaluation.

## Project overview

This project fine-tunes an ImageNet-pretrained **ResNet-50** for binary image classification. The full workflow is contained in [`Model_B_Mix.ipynb`](Model_B_Mix.ipynb): loading the image dataset, creating a stratified split, training the model, evaluating ROC AUC and accuracy, and testing robustness to common image transformations.

The model uses 224 × 224 RGB images and predicts one of two classes:

| Label | Meaning |
| --- | --- |
| `FAKE` | AI-generated image |
| `REAL` | Real image |

### Results snapshot

On the CIFAKE evaluation split, the initial ResNet-50 run achieved:

| Metric | Result |
| --- | ---: |
| Training accuracy (epoch 1) | 93.53% |
| Validation accuracy | 96.50% |
| Validation ROC AUC | 0.9955 |
| Clean test accuracy | 96.54% |
| Clean test ROC AUC | 0.9952 |

The notebook also evaluates robustness to JPEG compression, blur, resizing, noise, colour adjustment, and cropping. The final mixed-data model reached a clean SID ROC AUC of **0.9057** and an average corruption ROC AUC of **0.8746**; its weakest reported condition was resize at 0.25× (ROC AUC 0.8310).

## Repository structure

```text
.
├── Model_B_Mix.ipynb   # End-to-end experimentation, training, and evaluation
├── archive.zip         # Dataset archive (Git LFS-managed)
└── README.md
```

## Setup and installation

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook or Google Colab
- Git and Git LFS (required to retrieve the dataset archive)
- A CUDA-capable GPU is recommended; the notebook will fall back to CPU, though training will be substantially slower

### 1. Clone the repository and fetch the data

```bash
git clone https://github.com/koowenqi/TechJam-Ctrl-Alt-Elite.git
cd TechJam-Ctrl-Alt-Elite
git lfs install
git lfs pull
```

Confirm that `archive.zip` is a real ZIP archive rather than the small Git LFS pointer file before running the notebook.

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision scikit-learn pillow numpy tqdm jupyter
```

For a GPU-enabled PyTorch installation, use the command appropriate for your CUDA version from the [official PyTorch installer](https://pytorch.org/get-started/locally/).

### 3. Launch the notebook

```bash
jupyter notebook Model_B_Mix.ipynb
```

Google Colab is also supported. Upload or clone the repository there, then run the cells in order. The notebook contains a Colab-specific `files.download` step; omit or adapt that step when running locally.

## Steps to reproduce the results

1. Complete the setup above and ensure `archive.zip` is present.
2. Open `Model_B_Mix.ipynb` and select a GPU runtime if available.
3. Run the data-preparation cells. They extract the archive to `training/` and expect this layout:

   ```text
   training/
   ├── train/
   │   ├── FAKE/
   │   └── REAL/
   └── test/
       ├── FAKE/
       └── REAL/
   ```

4. Run the split cells. The notebook combines the provided train and test folders, then creates a stratified **70% / 15% / 15%** train/validation/test split with `SEED = 2026`.
5. Run the model setup and training cells. They initialise ImageNet-pretrained ResNet-50, replace its final layer with a two-class head, and train using AdamW (`lr=1e-4`, `weight_decay=1e-4`, batch size 64).
6. Run the validation and test cells to calculate accuracy and ROC AUC. The model checkpoint and split indices are saved in the Colab runtime by default; update those paths if running locally.
7. Run the corruption and external-evaluation sections to reproduce the robustness results. Exact scores can vary slightly with hardware and library versions.

## Limitations and future improvements

The strong clean-set score should be interpreted carefully: it may partly reflect dataset-specific artefacts rather than general AI-image detection ability. Performance falls under some transformations, particularly aggressive resizing, blur, and noise, which indicates sensitivity to low-level visual cues. The notebook also includes exploratory cells and is not yet packaged as a single command-line training pipeline.

With more time, we would:

- Evaluate on more diverse, independently collected datasets and unseen image generators.
- Train across a broader set of realistic post-processing augmentations.
- Run multi-epoch hyperparameter tuning and compare multiple backbones or an ensemble.
- Add calibration, per-class error analysis, and a documented held-out benchmark protocol.
- Refactor the notebook into reusable scripts, pin dependencies, and save artefacts within the repository or a release.

## Team contributions

| Team member | Contribution |
| --- | --- |
| Siah Siang Woo (Leader) | Collaborated on all aspects of the project: dataset preparation, model development, evaluation, robustness testing, and documentation |
| Ethan Cheang | Collaborated on all aspects of the project: dataset preparation, model development, evaluation, robustness testing, and documentation |
| Loke Kai Fong | Collaborated on all aspects of the project: dataset preparation, model development, evaluation, robustness testing, and documentation |
| Koo Wen Qi | Collaborated on all aspects of the project: dataset preparation, model development, evaluation, robustness testing, and documentation |

## Notes

- The dataset archive is stored through Git LFS; a normal `git clone` alone may only retrieve a pointer file.
- This repository is intended for experimentation and evaluation. It should not be used as the sole basis for high-stakes authenticity decisions.
