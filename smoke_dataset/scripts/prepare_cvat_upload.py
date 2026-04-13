from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten extracted frame folders into a CVAT-ready upload directory with unique filenames."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing extracted frames. Search is recursive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where flattened images and a manifest CSV will be written.",
    )
    return parser.parse_args()


def sanitize_part(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "root"


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_output_name(input_dir: Path, image_path: Path) -> str:
    relative_path = image_path.relative_to(input_dir)
    prefix_parts = [sanitize_part(part) for part in relative_path.parts[:-1]]
    prefix = "__".join(prefix_parts)
    filename = sanitize_part(image_path.stem) + image_path.suffix.lower()
    return f"{prefix}__{filename}" if prefix else filename


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "upload_manifest.csv"

    used_names: set[str] = set()
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_image", "relative_image", "upload_image"],
        )
        writer.writeheader()

        for image_path in images:
            output_name = build_output_name(input_dir, image_path)
            if output_name in used_names:
                raise RuntimeError(f"Duplicate upload filename generated: {output_name}")
            used_names.add(output_name)

            target_path = output_dir / output_name
            shutil.copy2(image_path, target_path)
            writer.writerow(
                {
                    "source_image": str(image_path),
                    "relative_image": str(image_path.relative_to(input_dir)),
                    "upload_image": output_name,
                }
            )

    print(f"Images copied: {len(images)}")
    print(f"Upload dir: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
