import argparse
import json
import os
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a NeRF-style transforms.json into transforms_train.json/transforms_test.json with per-frame time."
    )
    parser.add_argument("scene_dir", help="Scene directory containing transforms.json")
    parser.add_argument(
        "--input",
        default="transforms.json",
        help="Input transforms JSON filename relative to scene_dir",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="Seconds represented by one filename index step. Example: 0.5 for 2 FPS frames.",
    )
    parser.add_argument(
        "--zero-base",
        action="store_true",
        help="Subtract the minimum timestamp so the first frame starts at time 0.",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=8,
        help="Use every Nth frame for test split. Set to 0 or 1 to disable holdout.",
    )
    parser.add_argument(
        "--train-output",
        default="transforms_train.json",
        help="Output train transforms filename relative to scene_dir",
    )
    parser.add_argument(
        "--test-output",
        default="transforms_test.json",
        help="Output test transforms filename relative to scene_dir",
    )
    return parser.parse_args()


def frame_time_from_path(file_path: str, time_step: float) -> float:
    stem = Path(file_path).stem
    try:
        frame_index = int(stem)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse integer frame index from '{file_path}'. Expected names like 0001.png."
        ) from exc
    return frame_index * time_step


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    input_path = scene_dir / args.input
    train_output_path = scene_dir / args.train_output
    test_output_path = scene_dir / args.test_output

    with open(input_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {input_path}")

    augmented_frames = []
    for frame in frames:
        new_frame = deepcopy(frame)
        new_frame["time"] = frame_time_from_path(new_frame["file_path"], args.time_step)
        augmented_frames.append(new_frame)

    augmented_frames.sort(key=lambda frame: (frame["time"], frame["file_path"]))

    if args.zero_base:
        min_time = augmented_frames[0]["time"]
        for frame in augmented_frames:
            frame["time"] -= min_time

    if args.hold is not None and args.hold > 1:
        test_frames = [frame for idx, frame in enumerate(augmented_frames) if idx % args.hold == 0]
        train_frames = [frame for idx, frame in enumerate(augmented_frames) if idx % args.hold != 0]
    else:
        train_frames = augmented_frames
        test_frames = augmented_frames

    if not train_frames:
        raise ValueError("Train split is empty. Reduce --hold or check input frames.")
    if not test_frames:
        raise ValueError("Test split is empty. Reduce --hold or check input frames.")

    train_json = deepcopy(transforms)
    train_json["frames"] = train_frames

    test_json = deepcopy(transforms)
    test_json["frames"] = test_frames

    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f, indent=2)
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2)

    print(f"Wrote {len(train_frames)} train frames to {train_output_path}")
    print(f"Wrote {len(test_frames)} test frames to {test_output_path}")
    print(
        f"Time range: {train_frames[0]['time']:.6f} to {train_frames[-1]['time']:.6f} (train split, before loader normalization)"
    )


if __name__ == "__main__":
    main()
