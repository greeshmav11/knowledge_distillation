"""
Generates a figure grid of actual CIFAR-100 test images that illustrate KD's
effect, for use in the report / README (optional visual supplement to the
per-class delta bar chart).

Two grids are produced:
  1. "improved" examples: images the plain student got WRONG that the KD
     student got RIGHT, sampled from the classes with the largest *observed*
     (uncorrected) positive delta: mountain, seal, telephone, worm.
  2. "regressed" examples (optional, --include_regressed): the reverse, for
     the class with the largest observed (uncorrected) negative delta:
     flatfish.

IMPORTANT CAVEAT: these classes are picked from raw (uncorrected) per-class
McNemar tests in significance_analysis.json. None of them survive
Benjamini-Hochberg correction for testing 100 classes at once
(n_significant_after_bh_correction == 0). So these images illustrate the
largest *observed* differences between the plain and KD students, not a
statistically confirmed per-class effect. Don't caption these as "KD is
significantly better/worse at X" -- say "largest observed difference"
instead.

Uses results_50/raw_arrays.npz and results_50/improved_indices.npy /
regressed_indices.npy, already saved by analyze.py -- no re-inference needed.
Loads raw images directly via torchvision.datasets.CIFAR100 in the same
(unshuffled) index order analyze.py's test_loader used.

CONFIRMED (checked against data.py): get_cifar100_loaders builds the test
DataLoader with shuffle=False, so the index order used by analyze.py's
get_all_logits matches this script's direct torchvision.datasets.CIFAR100
iteration order. If you ever change the test loader to shuffle, this
alignment breaks and needs re-checking.

Usage:
    python src/make_example_grid.py --results_dir results_50/ --data_root ./data
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100

CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]
NAME_TO_IDX = {n: i for i, n in enumerate(CIFAR100_FINE_LABELS)}

# From significance_analysis.json's per-class (uncorrected) McNemar tests:
# these are the classes with p<0.05 BEFORE Benjamini-Hochberg correction.
# NONE of them survive BH correction (n_significant_after_bh_correction: 0),
# so treat these as "largest observed difference" examples, not confirmed
# per-class effects. Re-derive this list from significance_analysis.json
# if you re-run the experiment -- don't hand-edit without checking the JSON.
OBSERVED_TOP_HELPED = ["mountain", "seal", "telephone", "worm"]
OBSERVED_TOP_HURT = ["flatfish"]


def build_grid(dataset, flip_indices, labels, preds_plain, preds_kd,
                target_class_names, n_per_class, title, out_path):
    target_idxs = [NAME_TO_IDX[n] for n in target_class_names]
    flip_set = set(flip_indices.tolist())

    # collect up to n_per_class example indices per target class, restricted
    # to the flip set (i.e. examples where plain/kd actually disagreed)
    examples = {}
    for c in target_idxs:
        class_flip_idxs = [i for i in np.where(labels == c)[0] if i in flip_set]
        examples[c] = class_flip_idxs[:n_per_class]

    n_rows = len(target_idxs)
    n_cols = max((len(v) for v in examples.values()), default=1)
    n_cols = max(n_cols, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, c in enumerate(target_idxs):
        class_name = CIFAR100_FINE_LABELS[c]
        idxs = examples[c]
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idxs):
                idx = idxs[col]
                img, _ = dataset[idx]  # PIL image
                ax.imshow(img)
                plain_pred = CIFAR100_FINE_LABELS[preds_plain[idx]]
                kd_pred = CIFAR100_FINE_LABELS[preds_kd[idx]]
                ax.set_title(f"plain→{plain_pred}\nkd→{kd_pred}", fontsize=7)
            if col == 0:
                ax.text(-0.3, 0.5, class_name, fontsize=9, fontweight="bold",
                         rotation=90, va="center", ha="center",
                         transform=ax.transAxes)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_50")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_per_class", type=int, default=4)
    p.add_argument("--include_regressed", action="store_true",
                    help="also generate the grid of KD-hurt examples")
    args = p.parse_args()

    data = np.load(os.path.join(args.results_dir, "raw_arrays.npz"))
    labels = data["labels"]
    preds_plain = data["preds_plain"]
    preds_kd = data["preds_kd"]
    improved_indices = np.load(os.path.join(args.results_dir, "improved_indices.npy"))

    # train=False, download=True is safe -- torchvision skips download if
    # the tarball/extracted files are already present under data_root
    dataset = CIFAR100(root=args.data_root, train=False, download=True)

    build_grid(
        dataset, improved_indices, labels, preds_plain, preds_kd,
        OBSERVED_TOP_HELPED, args.n_per_class,
        title="Largest observed (uncorrected) KD gains: plain wrong -> KD right\n"
              "(mountain, seal, telephone, worm -- none survive BH correction)",
        out_path=os.path.join(args.results_dir, "example_grid_improved.png"),
    )

    if args.include_regressed:
        regressed_indices = np.load(os.path.join(args.results_dir, "regressed_indices.npy"))
        build_grid(
            dataset, regressed_indices, labels, preds_plain, preds_kd,
            OBSERVED_TOP_HURT, args.n_per_class,
            title="Largest observed (uncorrected) KD regression: plain right -> KD wrong\n"
                  "(flatfish -- does not survive BH correction)",
            out_path=os.path.join(args.results_dir, "example_grid_regressed.png"),
        )


if __name__ == "__main__":
    main()