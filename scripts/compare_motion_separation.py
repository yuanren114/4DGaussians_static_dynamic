import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def run_command(cmd, log_path, cwd):
    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write("COMMAND: {}\n\n".format(" ".join(cmd)))
        log_file.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        with open(log_path, "r") as log_file:
            tail = log_file.readlines()[-80:]
        raise RuntimeError("Command failed with exit code {}:\n{}\nLast log lines:\n{}".format(proc.returncode, " ".join(cmd), "".join(tail)))
    return elapsed


def load_results(model_path):
    results_path = Path(model_path) / "results.json"
    if not results_path.exists():
        return {}
    with open(results_path, "r") as results_file:
        results = json.load(results_file)
    if not results:
        return {}
    method_metrics = next(iter(results.values()))
    if not method_metrics:
        return {}
    return next(iter(method_metrics.values()))


def latest_render_dir(model_path):
    test_dir = Path(model_path) / "test"
    if not test_dir.exists():
        return None
    candidates = [p for p in test_dir.iterdir() if p.is_dir() and p.name.startswith("ours_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: int(p.name.split("_")[-1]))[-1]


def read_images(path):
    images = []
    for image_path in sorted(Path(path).glob("*.png")):
        images.append(np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0)
    return images


def laplacian_variance(image):
    gray = image.mean(axis=2)
    lap = -4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def proxy_metrics(model_path):
    render_dir = latest_render_dir(model_path)
    if render_dir is None:
        return {}
    renders = read_images(render_dir / "renders")
    if not renders:
        return {}
    stack = np.stack(renders, axis=0)
    h, w = stack.shape[1:3]
    border = max(1, min(h, w) // 10)
    border_pixels = np.concatenate([
        stack[:, :border, :, :].reshape(stack.shape[0], -1, 3),
        stack[:, -border:, :, :].reshape(stack.shape[0], -1, 3),
        stack[:, :, :border, :].reshape(stack.shape[0], -1, 3),
        stack[:, :, -border:, :].reshape(stack.shape[0], -1, 3),
    ], axis=1)
    center = stack[:, border:h-border, border:w-border, :] if h > 2 * border and w > 2 * border else stack
    return {
        "background_stability_proxy_border_temporal_variance": float(border_pixels.var(axis=0).mean()),
        "dynamic_sharpness_proxy_center_laplacian_variance": float(np.mean([laplacian_variance(img) for img in center])),
    }


def motion_mask_summary(model_path):
    stats_path = Path(model_path) / "motion_mask_stats.jsonl"
    if not stats_path.exists():
        return {}
    rows = []
    with open(stats_path, "r") as stats_file:
        for line in stats_file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "motion_mask_mean": last.get("mean"),
        "motion_mask_std": last.get("std"),
        "motion_mask_static_fraction": last.get("static_fraction"),
        "motion_mask_dynamic_fraction": last.get("dynamic_fraction"),
    }


def write_metrics(outputs_dir, rows):
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "metrics.json", "w") as json_file:
        json.dump(rows, json_file, indent=2)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(outputs_dir / "metrics.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_bars(outputs_dir, rows, names):
    plot_dir = outputs_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "PSNR", "SSIM", "LPIPS-vgg", "LPIPS-alex",
        "train_time_sec", "render_time_sec",
        "background_stability_proxy_border_temporal_variance",
        "dynamic_sharpness_proxy_center_laplacian_variance",
    ]
    for metric in metrics:
        values = [row.get(metric) for row in rows]
        if any(value is None for value in values):
            continue
        plt.figure(figsize=(5, 4))
        plt.bar(names, values)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(plot_dir / "{}.png".format(metric.replace("/", "_")))
        plt.close()

    motion_path = Path(rows[-1]["model_path"]) / "motion_mask_stats.jsonl"
    if motion_path.exists():
        values = []
        with open(motion_path, "r") as stats_file:
            for line in stats_file:
                if line.strip():
                    values.append(json.loads(line)["mean"])
        if values:
            plt.figure(figsize=(5, 4))
            plt.plot(values)
            plt.xlabel("logged step")
            plt.ylabel("motion mask mean")
            plt.tight_layout()
            plt.savefig(plot_dir / "motion_mask_mean_over_training.png")
            plt.close()


def write_summary(outputs_dir, rows):
    lines = [
        "# Baseline vs Motion Separation Summary",
        "",
        "Standard reconstruction metrics are produced by `metrics.py`: PSNR, SSIM, and LPIPS.",
        "Background stability and dynamic sharpness are proxy diagnostics, not formal benchmarks.",
        "",
    ]
    for row in rows:
        lines.append("## {}".format(row["variant"]))
        for key in sorted(row.keys()):
            if key not in {"variant", "model_path"}:
                lines.append("- {}: {}".format(key, row[key]))
        lines.append("- model_path: {}".format(row["model_path"]))
        lines.append("")
    with open(outputs_dir / "summary.md", "w") as summary_file:
        summary_file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--configs", default="")
    parser.add_argument("--expname", default="")
    parser.add_argument("--output-root", default="outputs/comparison")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--coarse-iterations", type=int, default=3000)
    parser.add_argument("--motion-mask-lambda", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    dataset_name = args.expname or Path(args.source_path).name
    outputs_dir = cwd / args.output_root / dataset_name
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline", []),
        ("motion_separation", ["--motion-separation", "--motion-mask-lambda", str(args.motion_mask_lambda)]),
    ]
    rows = []
    for variant, extra in variants:
        model_path = outputs_dir / variant
        common = [
            "-s", args.source_path,
            "--model_path", str(model_path),
            "--expname", "{}/{}".format(dataset_name, variant),
            "--iterations", str(args.iterations),
            "--coarse-iterations", str(args.coarse_iterations),
            "--save_iterations", str(args.iterations),
            "--test_iterations", str(args.iterations),
        ]
        if args.configs:
            common.extend(["--configs", args.configs])

        train_time = None
        render_time = None
        if not args.skip_train:
            train_time = run_command([sys.executable, "train.py"] + common + extra, logs_dir / "{}_train.log".format(variant), cwd)
        if not args.skip_render:
            render_cmd = [sys.executable, "render.py", "--model_path", str(model_path), "--skip_train", "--skip_video"]
            if args.configs:
                render_cmd.extend(["--configs", args.configs])
            render_time = run_command(render_cmd, logs_dir / "{}_render.log".format(variant), cwd)
        if not args.skip_metrics:
            run_command([sys.executable, "metrics.py", "--model_paths", str(model_path)], logs_dir / "{}_metrics.log".format(variant), cwd)

        row = {"variant": variant, "model_path": str(model_path)}
        row.update(load_results(model_path))
        row.update(proxy_metrics(model_path))
        row.update(motion_mask_summary(model_path))
        if train_time is not None:
            row["train_time_sec"] = train_time
        if render_time is not None:
            row["render_time_sec"] = render_time
        rows.append(row)

    write_metrics(outputs_dir, rows)
    plot_bars(outputs_dir, rows, [name for name, _ in variants])
    write_summary(outputs_dir, rows)
    print("Wrote comparison outputs to {}".format(outputs_dir))


if __name__ == "__main__":
    main()
