"""
Parses the raw stdout logs produced by train.py (e.g. logs/teacher.log,
logs/student_plain.log, logs/student_kd.log) and plots train_loss and
test_acc curves for your slides.

Expects lines like:
  [teacher] epoch 1/60  train_loss=4.4047  test_acc=0.0836  best=0.0836  (29.2s)

Produces, per model found:
  - <out_dir>/<model>_curves.png   (train_loss + test_acc, two panels)
And one combined figure:
  - <out_dir>/all_models_test_acc.png   (test_acc for all models overlaid,
    useful for a single "convergence" slide comparing all three)

Usage:
    python plot_training_curves.py --log_dir logs --out_dir results
    # or point at specific files:
    python plot_training_curves.py --logs logs/teacher.log logs/student_plain.log logs/student_kd.log --out_dir results
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"\[(?P<model>[\w_]+)\]\s+epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
    r"train_loss=(?P<train_loss>[\d.]+)\s+test_acc=(?P<test_acc>[\d.]+)\s+"
    r"best=(?P<best>[\d.]+)"
)


def parse_log_file(path):
    """Returns dict: model_name -> {'epoch': [...], 'train_loss': [...],
    'test_acc': [...], 'best': [...]}. A single log file may contain lines
    for more than one model tag, though typically one file = one model."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            model = m.group("model")
            d = data.setdefault(model, {"epoch": [], "train_loss": [], "test_acc": [], "best": []})
            d["epoch"].append(int(m.group("epoch")))
            d["train_loss"].append(float(m.group("train_loss")))
            d["test_acc"].append(float(m.group("test_acc")))
            d["best"].append(float(m.group("best")))
    return data


def plot_single_model(model_name, d, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(d["epoch"], d["train_loss"], color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.set_title(f"{model_name}: training loss")
    ax1.grid(alpha=0.3)

    ax2.plot(d["epoch"], d["test_acc"], color="tab:blue", label="test_acc (this epoch)")
    ax2.plot(d["epoch"], d["best"], color="tab:blue", linestyle="--", alpha=0.6, label="best so far")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    ax2.set_title(f"{model_name}: test accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Training curves: {model_name}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_combined_test_acc(all_data, out_dir):
    """One figure, test_acc for every model overlaid -- good for a single
    'all three models converged' slide."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"teacher": "tab:green", "student_plain": "tab:orange", "student_kd": "tab:purple"}
    for model_name, d in all_data.items():
        color = colors.get(model_name)
        ax.plot(d["epoch"], d["test_acc"], label=model_name, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Test accuracy convergence, all models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "all_models_test_acc.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default=None,
                    help="Directory to glob for *.log files (e.g. 'logs')")
    p.add_argument("--logs", nargs="*", default=None,
                    help="Explicit list of log file paths, alternative to --log_dir")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    if not args.log_dir and not args.logs:
        raise SystemExit("Provide either --log_dir or --logs")

    log_paths = []
    if args.log_dir:
        log_paths.extend(sorted(glob.glob(os.path.join(args.log_dir, "*.log"))))
    if args.logs:
        log_paths.extend(args.logs)
    log_paths = sorted(set(log_paths))

    if not log_paths:
        raise SystemExit(f"No log files found (looked in: {args.log_dir}, {args.logs})")

    os.makedirs(args.out_dir, exist_ok=True)

    all_data = {}
    for path in log_paths:
        parsed = parse_log_file(path)
        if not parsed:
            print(f"WARNING: no matching lines found in {path} -- check the log "
                  f"format matches train.py's print format.")
            continue
        for model_name, d in parsed.items():
            if model_name in all_data:
                print(f"NOTE: model '{model_name}' appears in multiple log files; "
                      f"using the version from {path}.")
            all_data[model_name] = d

    if not all_data:
        raise SystemExit("Parsed 0 models from the given log files -- nothing to plot.")

    written = []
    for model_name, d in all_data.items():
        out_path = plot_single_model(model_name, d, args.out_dir)
        written.append(out_path)
        print(f"{model_name}: {len(d['epoch'])} epochs parsed -> {out_path}")

    if len(all_data) > 1:
        combined_path = plot_combined_test_acc(all_data, args.out_dir)
        written.append(combined_path)
        print(f"Combined test_acc plot -> {combined_path}")

    print("\nDone. Files written:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()