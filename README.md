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

# AIGC (AI-Generated Content) Image Detector Script `predict.py`

This script runs the trained **Model B-Mix** (ResNet50, fine-tuned on SID_Set +
CIFAKE with mixed-domain rehearsal) over a folder of images from a dir and outputs a
confidence score for each one, indicating the likelihood that it is
AI-generated content (1 being high confidence that it is an AI image)

It depends on the trained checkpoint file produced by main file (model.pth)

---

##  Requirements

- Python 3.9+
- The following packages:
  ```bash
  pip install torch torchvision pillow tqdm
  ```
---

##  Necessities

| Item | Description |
|---|---|
| `predict.py` | This script. |
| A trained checkpoint | `resnet50_model_b_mix_best.pth`, produced and downloaded from the training notebook. |
| A folder of images | Any folder containing `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, or `.tiff` files. Subfolders are scanned automatically. |

You do **not** need to run the training notebook again if you already have
the `.pth` checkpoint saved somewhere. The script rebuilds the model
architecture and loads the trained weights from that file directly

---

##  Usage

Basic command:

```bash
python predict.py --image_dir "/path/to/your/images" --checkpoint "path/to/resnet50_model_b_mix_best.pth" --output "path/to/predictions.json"
```

This creates `predictions.json` in the specified folder by default.

### Full set of options

| Flag | Required | Default | Description |
|---|---|---|---|
| `--image_dir` | ✅ | — | Folder of images to score (searched recursively). |
| `--checkpoint` | ✅ | — | Path to the trained `.pth` checkpoint file. |
| `--output` | ❌ | `predictions.json` | Where to write the results. **Use a full/absolute path** — see note below. |
| `--batch_size` | ❌ | `32` | Number of images processed per batch. |
| `--num_workers` | ❌ | `2` | Parallel data-loading workers. |
| `--device` | ❌ | auto-detect | Force `cpu` or `cuda`. |

### Example (Windows, full paths)

```powershell
python "Z:\path\to\predict.py" ^
  --image_dir "Z:\path\to\testing" ^
  --checkpoint "Z:\path\to\resnet50_model_b_mix_best.pth" ^
  --output "Z:\path\to\predictions.json"
```

> **Always pass a full/absolute path to `--output`.** A relative path like
> `predictions.json` is created in your terminal's *current working
> directory*, not necessarily the folder your script or images are in — this
> is a common source of confusion (and of "file not found" surprises
> afterward).

---

##  Output format

The script produces a single JSON file: a list of objects, one per image,
each with:

```json
[
  {
    "image_path": "/path/to/images/photo1.jpg",
    "pred": 0.8213
  },
  {
    "image_path": "/path/to/images/photo2.png",
    "pred": 0.0421
  }
]
```

- `image_path` — full path to the image that was scored.
- `pred` — a float between `0.0` and `1.0`: the model's estimated probability
  that the image is AI-generated. Values closer to `1.0` indicate higher
  confidence the image is AIGC; values closer to `0.0` indicate higher
  confidence the image is authentic/real.

There is no fixed "AI vs Real" cutoff baked into the script.
It reports raw probabilities so results can be thresholded or reviewed as needed.

---

##  Preprocessing details

Images are preprocessed identically to how the model was validated during
training:

1. Resize shortest side to 256px
2. Center-crop to 224×224
3. Convert to tensor
4. Normalize using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`,
   `std=[0.229, 0.224, 0.225]`)

---

##  Troubleshooting

**`PermissionError: [Errno 13] Permission denied: 'predictions.json'`**
Almost always means the script can't write to the target folder — not a
problem with the file itself (the script creates `predictions.json`
automatically; you never need to create it beforehand). Common fixes:
- Pass a full/absolute path to `--output` rather than a bare filename.
- If working on a mapped network drive (e.g. `Z:\...` on a university
  network), try writing locally instead (e.g. your Desktop) to confirm
  whether the drive itself is read-restricted for your account.
- Make sure no other program has `predictions.json` open if one already
  exists at that path.

**`FileNotFoundError` for the checkpoint**
Double-check the `--checkpoint` path points directly at the `.pth` file
(not just its containing folder), and that the file was fully downloaded
from the training notebook (checkpoints are typically ~100MB for ResNet50 —
if the file is much smaller, the download may be incomplete or corrupted).

**No images found**
Confirm `--image_dir` points at a folder that actually contains image files
with one of the supported extensions, and that you have read access to it.


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
