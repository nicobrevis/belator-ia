from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def sort_key(value: str) -> tuple[int, str]:
    head = value.split("_", 1)[0]
    return (int(head), value) if head.isdigit() else (10**9, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a stable Ultralytics YOLO segmentation dataset from a CVAT export. "
            "Creates train/val splits grouped by source video prefix."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Root of an exported dataset from CVAT. "
            "Pass the argument multiple times to merge several exports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the prepared dataset will be created.",
    )
    parser.add_argument(
        "--val-groups",
        nargs="+",
        required=True,
        help=(
            "Video groups to reserve for validation. "
            "The group is the filename prefix before '_f', for example '8' in '8_f000030...jpg'."
        ),
    )
    return parser.parse_args()


def group_name(image_path: Path) -> str:
    return image_path.stem.split("_f", 1)[0]


def write_split_list(root: Path, split: str) -> None:
    images = sorted((root / "images" / split).glob("*.jpg"))
    output = root / f"{split}.txt"
    with output.open("w", encoding="utf-8") as f:
        for image in images:
            f.write(f"images/{split}/{image.name}\n")


def main() -> None:
    args = parse_args()
    source_dirs = [path.resolve() for path in args.source_dir]
    output_dir = args.output_dir
    val_groups = set(args.val_groups)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    positive_train = 0
    positive_val = 0
    seen_names: set[str] = set()

    for source_dir in source_dirs:
        source_images = source_dir / "images" / "train"
        source_labels = source_dir / "labels" / "train"

        if not source_images.exists():
            raise FileNotFoundError(f"Missing source images folder: {source_images}")
        if not source_labels.exists():
            raise FileNotFoundError(f"Missing source labels folder: {source_labels}")

        image_paths = sorted(source_images.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {source_images}")

        for image_path in image_paths:
            if image_path.name in seen_names:
                raise RuntimeError(f"Duplicate image name across exports: {image_path.name}")
            seen_names.add(image_path.name)

            split = "val" if group_name(image_path) in val_groups else "train"
            destination_image = output_dir / "images" / split / image_path.name
            shutil.copy2(image_path, destination_image)

            label_path = source_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                destination_label = output_dir / "labels" / split / label_path.name
                shutil.copy2(label_path, destination_label)
                if split == "train":
                    positive_train += 1
                else:
                    positive_val += 1

            if split == "train":
                train_count += 1
            else:
                val_count += 1

    write_split_list(output_dir, "train")
    write_split_list(output_dir, "val")

    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("names:\n")
        f.write("  0: smoke\n")

    print(f"Prepared dataset: {output_dir}")
    print(f"Source exports: {', '.join(str(path) for path in source_dirs)}")
    print(f"Validation groups: {', '.join(sorted(val_groups, key=sort_key))}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Train positives: {positive_train}")
    print(f"Val positives: {positive_val}")
    print(f"Train negatives: {train_count - positive_train}")
    print(f"Val negatives: {val_count - positive_val}")


if __name__ == "__main__":
    main()
