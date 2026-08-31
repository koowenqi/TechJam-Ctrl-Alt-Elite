"""
RUN MODEL FIRST to get model.pth file to run script
Run line below in powershell (change dir to those you are using)
python predict.py --image_dir "/path/to/your/images" --checkpoint "path/to/resnet50_model_b_mix_best.pth" --output "path/to/predictions.json"
check json file in vsc for predictions
(if pred value closer to 1.0, means more likely to be AI/probability of AI)
"""
# imports
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# Config

# File extensions accepted 
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Preprocessing pipeline (sid_transform / cifake_clean_transform / clean_validation_transform)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet channel means
        std=[0.229, 0.224, 0.225],   # ImageNet channel stds
    ),
])


# loads and preprocesses every image in a directory

class ImageFolderDataset(Dataset):
    """
    Collects every image file under `root_dir` and serves (transformed_tensor, image_path) pairs. 
    Corrupt/unreadable files are skipped
    """

    def __init__(self, root_dir: Path, transform):
        self.transform = transform
        self.image_paths = []

        candidate_paths = sorted(
            p for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for path in candidate_paths:
            # Verify the file can actually be opened
            try:
                with Image.open(path) as img:
                    img.verify()
                self.image_paths.append(path)
            except Exception as error:
                print(f"[WARN] Skipping unreadable image {path}: {error}")

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found under: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")

        tensor = self.transform(image)

        # Return as a string so it can be batched
        return tensor, str(path)

# Load Model

def load_model(checkpoint_path: Path, device: torch.device):
    # loads trained weights + metadata from the checkpoint.

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # disables dropout/batchnorm updates for inference

    ai_class_index = checkpoint["ai_class_index"]

    return model, ai_class_index

# Run on image data

def run_inference(model, ai_class_index, data_loader, device):

    # Runs the model over every batch in `data_loader` and returns a list of dicts, where `pred` is the probability
    results = []

    with torch.no_grad():
        for images, paths in tqdm(data_loader, desc="Running inference"):
            images = images.to(device)

            outputs = model(images)                       # raw logits, shape (batch, 2)
            probabilities = torch.softmax(outputs, dim=1)  # convert to probabilities

            # Pull out just the AI class probability for each image
            ai_probabilities = probabilities[:, ai_class_index].tolist()

            for path, pred in zip(paths, ai_probabilities):
                results.append({
                    "image_path": path,
                    "pred": round(float(pred), 6),
                })

    return results


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Score a directory of images for likelihood of being AI-generated."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing images to score (searched recursively).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="resnet50_model_b_mix_best.pth",
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.json",
        help="Where to write the output JSON file.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force 'cpu' or 'cuda'. Defaults to cuda if available, else cpu.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    image_dir = Path(args.image_dir)
    checkpoint_path = Path(args.checkpoint)

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir does not exist or is not a directory: {image_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # Build dataset + loader
    dataset = ImageFolderDataset(image_dir, INFERENCE_TRANSFORM)
    print(f"Found {len(dataset)} valid images under {image_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          
                                
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load model + run predictions
    model, ai_class_index = load_model(checkpoint_path, device)
    print(f"Loaded model. AI-generated class index: {ai_class_index}")

    results = run_inference(model, ai_class_index, loader, device)

    # Write JSON output
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
