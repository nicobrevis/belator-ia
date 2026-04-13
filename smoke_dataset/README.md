# Smoke Dataset Workflow

This folder is the starting point for building a smoke dataset from drone videos before annotation in CVAT.

## Recommended local structure

Place your source videos here:

- `smoke_dataset/data/raw_videos`

The extractor will write sampled frames here:

- `smoke_dataset/data/frames_full`
- `smoke_dataset/data/frames_horizon`

You can use this folder as a staging area before uploading to CVAT:

- `smoke_dataset/data/cvat_uploads`

And store exported annotations or final datasets here:

- `smoke_dataset/exports`

## Why your current YOLO setup is likely struggling

From the sample frame, the smoke plumes are:

- small relative to the full image,
- low-contrast,
- close to the horizon,
- visually similar to haze, clouds, dust, glare, and compression artifacts.

That combination usually causes low-confidence detections when training on full frames resized to 640 or 1024, because the smoke occupies too few useful pixels.

## What to do first

1. Extract frames from your flight videos instead of labeling raw videos end to end.
2. Keep both positive frames and hard negatives:
   cloud bands, haze, dust, fog, road glare, bright roofs, and smoke-like reflections.
3. Annotate smoke as polygons in CVAT for better shape supervision.
4. Re-train only after the first dataset pass is reasonably clean.

## Recommended extraction strategy

For distant smoke from drone footage, start with these settings:

- sample rate: `0.5` to `1.0` fps
- blur threshold: `80`
- dedupe threshold: `2.0`
- full frame first

If most smoke appears near the horizon, also generate a second dataset version cropped to the upper-middle band. Example:

- `--roi-top-ratio 0.30`
- `--roi-bottom-ratio 0.75`

This gives you a horizon-focused dataset where the smoke occupies more pixels, which is often much easier for the model to learn.

## Extract frames

Example using full frames:

```bash
python smoke_dataset/scripts/extract_frames.py ^
  --input-dir smoke_dataset\data\raw_videos ^
  --output-dir smoke_dataset\data\frames_full ^
  --fps 0.5 ^
  --blur-threshold 80 ^
  --dedupe-threshold 2.0
```

Example using a horizon crop:

```bash
python smoke_dataset/scripts/extract_frames.py ^
  --input-dir smoke_dataset\data\raw_videos ^
  --output-dir smoke_dataset\data\frames_horizon ^
  --fps 1.0 ^
  --blur-threshold 80 ^
  --dedupe-threshold 2.0 ^
  --roi-top-ratio 0.30 ^
  --roi-bottom-ratio 0.75
```

The script writes:

- extracted `.jpg` frames grouped by video
- `frames_manifest.csv` with source video, frame index, timestamp, and blur score

## What to upload to CVAT

Do not upload the original videos first. Upload the extracted frames.

The usual flow is:

1. copy the short drone videos into `smoke_dataset/data/raw_videos`
2. run the extractor
3. review the extracted images in `smoke_dataset/data/frames_full` or `smoke_dataset/data/frames_horizon`
4. upload one of those frame folders into a CVAT task inside the project
5. annotate the frames with the `smoke` polygon label

## Annotation guidance

In CVAT, start simple:

- class: `smoke`
- shape: polygon

Consistency matters more than perfection. If the plume is diffuse, label the visible dense body of the smoke, not the entire uncertain haze around it.

## Training guidance

These are the highest-value improvements for your case:

1. Do not rely only on full-frame detection. Use crops, tiles, or horizon-band training.
2. Add many hard negatives on purpose. This is one of the biggest levers for false positives and low-confidence behavior.
3. Keep a validation set split by flight, not by random frame. Frames from the same video are too similar.
4. If deployment must be real-time, train on crops or tiles but evaluate on the real inference pipeline you will actually deploy.
5. Consider a two-stage pipeline:
   first candidate detection on horizon tiles, then temporal confirmation over consecutive frames.

## Suggested next milestone

Build a first dataset with:

- 500 to 1,500 extracted candidate frames
- at least 30 to 40 percent hard negatives
- positives from multiple flights, altitudes, weather conditions, and camera zoom levels

Once that exists, we can move to:

1. task creation in CVAT from extracted frames,
2. annotation protocol,
3. export to YOLO format,
4. training recipe for a first strong baseline.

## Training the first baseline

Prepared training dataset:

- `smoke_dataset/datasets/pyrone_172_v1`

Recommended first run on a GTX 1650 Ti 4 GB:

```bash
.\.venv-smoke\Scripts\python.exe smoke_dataset/scripts/train_smoke.py ^
  --data smoke_dataset/datasets/pyrone_172_v1/data.yaml ^
  --model yolo11n-seg.pt ^
  --epochs 80 ^
  --imgsz 960 ^
  --batch 2 ^
  --workers 2 ^
  --cache ^
  --run-val
```

`train_smoke.py` keeps AMP disabled by default for stability on this GPU.

If you want to try mixed precision later, add:

- `--amp`

If you get an out-of-memory error, lower:

- `--batch 2`
- or `--imgsz 768`

Artifacts will be saved under:

- `smoke_dataset/runs`

## Reviewing the trained model on raw videos

Use the prediction helper to render videos with detections and generate a per-video summary:

```bash
.\.venv-smoke\Scripts\python.exe smoke_dataset/scripts/predict_smoke.py ^
  --source smoke_dataset/data/raw_videos ^
  --model smoke_dataset/runs/pyrone_172_v1_yolo11n_seg_e80_i960_b2/weights/best.pt ^
  --imgsz 960 ^
  --conf 0.50 ^
  --vid-stride 5 ^
  --name pyrone_172_v1_all_videos_conf050_stride5
```

Outputs will be written under:

- `smoke_dataset/predictions`

The run folder includes:

- rendered `.mp4` videos with predicted masks
- `summary.csv` with processed frames, detection rate, total detections, and confidence stats per video
