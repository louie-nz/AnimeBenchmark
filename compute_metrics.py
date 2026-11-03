# Generate CLIP embeddings for all images in a folder and save as .npy files

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip


def iter_image_files(folder: str):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in img_exts:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for all images in a folder and save as .npy files."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Folder to save the .npy files. Defaults to --input_dir.",
    )
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="openai", help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch")
    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda; auto uses CUDA if available")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(iter_image_files(str(input_dir)))
    if not image_paths:
        raise FileNotFoundError(f"No images were found in: {input_dir}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model, _, preprocess = open_clip.create_model_and_preprocess(
        model_name=args.model,
        pretrained=args.pretrained,
    )
    model = model.to(device)
    model.eval()

    total = 0
    with torch.no_grad():
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i : i + args.batch_size]
            valid_paths = []
            batch_tensors = []

            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    tensor = preprocess(image)
                    batch_tensors.append(tensor)
                    valid_paths.append(image_path)
                except Exception as exc:
                    print(f"Skipping {image_path}: {exc}")

            if not valid_paths:
                continue

            batch = torch.stack(batch_tensors).to(device)
            embeddings = model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            for image_path, embedding in zip(valid_paths, embeddings):
                npy_path = output_dir / f"{image_path.stem}.npy"
                np.save(npy_path, embedding.cpu().numpy().astype(np.float32))
                print(f"Saved embedding for {image_path.name} -> {npy_path}")
                total += 1

    print(f"Done. Generated {total} CLIP embeddings in {output_dir}")


if __name__ == "__main__":
    main()
