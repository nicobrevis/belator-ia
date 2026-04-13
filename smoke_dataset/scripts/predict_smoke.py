from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smoke-segmentation inference on drone videos and save a review summary."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "smoke_dataset/runs/pyrone_172_v1_yolo11n_seg_e80_i960_b2/weights/best.pt"
        ),
        help="Path to the trained YOLO segmentation model.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("smoke_dataset/data/raw_videos"),
        help="Video file or directory to process.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("smoke_dataset/predictions"),
        help="Directory where prediction runs are stored.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pyrone_172_v1_best_review_conf015",
        help="Run name.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size. Default: 960",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold. Default: 0.15",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold. Default: 0.5",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame. Default: 1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device. Use 'auto', 'cpu', or a GPU id like '0'. Default: auto",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format predictions next to the rendered outputs.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in saved YOLO prediction text files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable Ultralytics per-frame logging.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def write_summary(summary_path: Path, per_video: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "video",
        "frames_processed",
        "frames_with_detections",
        "detection_frame_ratio",
        "total_detections",
        "mean_confidence",
        "max_confidence",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video_name in sorted(per_video, key=lambda name: int(Path(name).stem)):
            stats = per_video[video_name]
            frames = int(stats["frames_processed"])
            detections = int(stats["total_detections"])
            conf_count = int(stats["confidence_count"])
            writer.writerow(
                {
                    "video": video_name,
                    "frames_processed": frames,
                    "frames_with_detections": int(stats["frames_with_detections"]),
                    "detection_frame_ratio": (
                        round(stats["frames_with_detections"] / frames, 4) if frames else 0.0
                    ),
                    "total_detections": detections,
                    "mean_confidence": (
                        round(stats["confidence_sum"] / conf_count, 4) if conf_count else 0.0
                    ),
                    "max_confidence": round(stats["max_confidence"], 4),
                }
            )


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    source_path = args.source.resolve()
    project_dir = args.project.resolve()
    save_dir = project_dir / args.name
    project_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source not found: {source_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Save dir: {save_dir}")

    model = YOLO(str(model_path))
    per_video: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "frames_processed": 0.0,
            "frames_with_detections": 0.0,
            "total_detections": 0.0,
            "confidence_sum": 0.0,
            "confidence_count": 0.0,
            "max_confidence": 0.0,
        }
    )

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        vid_stride=args.vid_stride,
        retina_masks=True,
        project=str(project_dir),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        exist_ok=True,
        stream=True,
        verbose=args.verbose,
    )

    for result in results:
        video_name = Path(result.path).name
        stats = per_video[video_name]
        stats["frames_processed"] += 1

        det_count = 0 if result.boxes is None else len(result.boxes)
        stats["total_detections"] += det_count
        if det_count > 0:
            stats["frames_with_detections"] += 1
            confidences = result.boxes.conf.tolist()
            stats["confidence_sum"] += sum(confidences)
            stats["confidence_count"] += len(confidences)
            stats["max_confidence"] = max(stats["max_confidence"], max(confidences))

    summary_path = save_dir / "summary.csv"
    write_summary(summary_path, per_video)

    print("Prediction finished.")
    print(f"Rendered outputs: {save_dir}")
    print(f"Summary CSV: {summary_path}")
    print("Per-video overview:")
    for video_name in sorted(per_video, key=lambda name: int(Path(name).stem)):
        stats = per_video[video_name]
        frames = int(stats["frames_processed"])
        frames_with_det = int(stats["frames_with_detections"])
        total_dets = int(stats["total_detections"])
        mean_conf = (
            stats["confidence_sum"] / stats["confidence_count"]
            if stats["confidence_count"]
            else 0.0
        )
        print(
            f"- {video_name}: frames={frames}, detection_frames={frames_with_det}, "
            f"detections={total_dets}, mean_conf={mean_conf:.3f}, "
            f"max_conf={stats['max_confidence']:.3f}"
        )


if __name__ == "__main__":
    main()
