from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a smoke segmentation baseline with Ultralytics YOLO."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("smoke_dataset/datasets/pyrone_172_v1/data.yaml"),
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs. Default: 80",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Training image size. Default: 960",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size. Reduce to 2 if VRAM is not enough. Default: 4",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Data loader workers. Default: 2",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("smoke_dataset/runs"),
        help="Directory where Ultralytics stores runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pyrone_172_v1_yolo11n_seg",
        help="Run name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device. Use 'auto', 'cpu', or a GPU id like '0'. Default: auto",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience. Default: 20",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic during the last N epochs. Default: 10",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable dataset caching in RAM.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (disabled by default for stability on this GPU).",
    )
    parser.add_argument(
        "--run-val",
        action="store_true",
        help="Run a validation pass on the best checkpoint after training.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def build_resolved_yaml(data_yaml: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = data_yaml.parent.resolve()
    config["path"] = dataset_root.as_posix()

    temp_dir = Path(tempfile.gettempdir()) / "pyrone_ultralytics"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for split_key in ("train", "val", "test"):
        split_ref = config.get(split_key)
        if not isinstance(split_ref, str):
            continue

        split_path = (dataset_root / split_ref).resolve()
        if split_path.suffix.lower() == ".txt" and split_path.exists():
            resolved_list_path = temp_dir / f"{data_yaml.stem}.{split_key}.txt"
            resolved_lines = []
            for raw_line in split_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                item_path = Path(line)
                if not item_path.is_absolute():
                    item_path = (dataset_root / item_path).resolve()
                resolved_lines.append(item_path.as_posix())
            resolved_list_path.write_text(
                "\n".join(resolved_lines) + "\n",
                encoding="utf-8",
            )
            config[split_key] = resolved_list_path.as_posix()
        elif split_ref:
            config[split_key] = split_path.as_posix()

    resolved_yaml = temp_dir / f"{data_yaml.stem}.resolved.yaml"

    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    return resolved_yaml


def main() -> None:
    args = parse_args()

    data_yaml = args.data.resolve()
    project_dir = args.project.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    resolved_yaml = build_resolved_yaml(data_yaml)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Resolved dataset YAML: {resolved_yaml}")
    print(f"AMP enabled: {args.amp}")

    model = YOLO(args.model)
    results = model.train(
        data=str(resolved_yaml),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        device=device,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        cache=args.cache,
        pretrained=True,
        cos_lr=True,
        seed=42,
        deterministic=True,
        amp=args.amp,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    print(f"Training finished. Artifacts: {results.save_dir}")

    if args.run_val:
        best_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_path.exists():
            print(f"Running validation for: {best_path}")
            YOLO(str(best_path)).val(
                data=str(resolved_yaml),
                imgsz=args.imgsz,
                batch=1,
                device=device,
                split="val",
                plots=True,
            )
        else:
            print("Best checkpoint not found, skipping validation.")


if __name__ == "__main__":
    main()
