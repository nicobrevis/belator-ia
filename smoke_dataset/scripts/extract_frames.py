from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
}


@dataclass
class ExtractionStats:
    saved: int = 0
    skipped_blur: int = 0
    skipped_duplicate: int = 0
    sampled: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sampled frames from drone videos to build a smoke dataset. "
            "Supports blur filtering, near-duplicate filtering and optional ROI cropping."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source videos. Search is recursive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted frames and the manifest CSV will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="How many frames per second to sample from each video. Default: 0.5",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=80.0,
        help=(
            "Minimum Laplacian variance to keep a frame. Increase to reject more blurry frames. "
            "Default: 80.0"
        ),
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=2.0,
        help=(
            "Minimum mean absolute grayscale difference vs. the last kept frame. "
            "Lower keeps more similar frames. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--roi-top-ratio",
        type=float,
        default=0.0,
        help="Top crop ratio for the saved frame. Use 0.35 for an upper horizon crop. Default: 0.0",
    )
    parser.add_argument(
        "--roi-bottom-ratio",
        type=float,
        default=1.0,
        help="Bottom crop ratio for the saved frame. Use 0.75 for an upper horizon crop. Default: 1.0",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Optional cap of saved frames per video. Use 0 for no cap. Default: 0",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames. Default: 95",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if not 0.0 <= args.roi_top_ratio < args.roi_bottom_ratio <= 1.0:
        raise ValueError("--roi-top-ratio and --roi-bottom-ratio must satisfy 0 <= top < bottom <= 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in the range [1, 100]")
    if args.max_frames_per_video < 0:
        raise ValueError("--max-frames-per-video must be >= 0")


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def crop_roi(frame, top_ratio: float, bottom_ratio: float):
    height = frame.shape[0]
    top = int(height * top_ratio)
    bottom = int(height * bottom_ratio)
    return frame[top:bottom, :]


def blur_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)


def mean_abs_diff(a, b) -> float:
    diff = cv2.absdiff(a, b)
    return float(diff.mean())


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    manifest_writer: csv.DictWriter,
) -> ExtractionStats:
    stats = ExtractionStats()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    frame_step = max(1, round(source_fps / args.fps))
    frame_index = -1
    saved_index = 0
    previous_signature = None

    relative_parent = video_path.parent.relative_to(args.input_dir)
    target_dir = output_dir / relative_parent / video_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        stats.sampled += 1

        roi_frame = crop_roi(frame, args.roi_top_ratio, args.roi_bottom_ratio)
        current_blur = blur_score(roi_frame)
        if current_blur < args.blur_threshold:
            stats.skipped_blur += 1
            continue

        signature = frame_signature(roi_frame)
        if previous_signature is not None:
            diff = mean_abs_diff(signature, previous_signature)
            if diff < args.dedupe_threshold:
                stats.skipped_duplicate += 1
                continue

        previous_signature = signature
        timestamp_sec = frame_index / source_fps
        output_path = target_dir / f"{video_path.stem}_f{frame_index:06d}_t{timestamp_sec:09.2f}s.jpg"
        cv2.imwrite(
            str(output_path),
            roi_frame,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )

        stats.saved += 1
        saved_index += 1

        manifest_writer.writerow(
            {
                "source_video": str(video_path),
                "output_image": str(output_path),
                "relative_image": str(output_path.relative_to(output_dir)),
                "frame_index": frame_index,
                "time_seconds": f"{timestamp_sec:.2f}",
                "source_fps": f"{source_fps:.3f}",
                "width": roi_frame.shape[1],
                "height": roi_frame.shape[0],
                "blur_score": f"{current_blur:.2f}",
            }
        )

        if args.max_frames_per_video and saved_index >= args.max_frames_per_video:
            break

    cap.release()
    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    videos = find_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "frames_manifest.csv"

    total = ExtractionStats()
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "source_video",
                "output_image",
                "relative_image",
                "frame_index",
                "time_seconds",
                "source_fps",
                "width",
                "height",
                "blur_score",
            ],
        )
        writer.writeheader()

        for video_path in videos:
            stats = extract_from_video(video_path, args.output_dir, args, writer)
            total.saved += stats.saved
            total.sampled += stats.sampled
            total.skipped_blur += stats.skipped_blur
            total.skipped_duplicate += stats.skipped_duplicate
            print(
                f"[ok] {video_path.name}: saved={stats.saved} sampled={stats.sampled} "
                f"blur_skip={stats.skipped_blur} duplicate_skip={stats.skipped_duplicate}"
            )

    print()
    print(f"Videos processed: {len(videos)}")
    print(f"Frames sampled: {total.sampled}")
    print(f"Frames saved: {total.saved}")
    print(f"Skipped by blur: {total.skipped_blur}")
    print(f"Skipped by duplication: {total.skipped_duplicate}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
