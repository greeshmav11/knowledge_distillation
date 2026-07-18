"""
Robustness follow-up to analyze.py. Uses the arrays analyze.py already saved
(results/raw_arrays.npz) -- no re-inference, no retraining. Adds:

  1. Per-class McNemar's exact test (plain student vs KD student), paired on
     the same test images, to separate "really hurt/helped by KD" classes
     from classes whose delta is plausibly just n=100 sampling noise.
  2. Bootstrap confidence intervals (resampling test examples with
     replacement) for the headline scalar metrics: overall_acc deltas,
     mean_per_class_delta, and n_classes_hurt/helped (recomputed each
     resample so the CI reflects real sampling variability, not just a
     point estimate).
  3. Pairwise confusion analysis for a configurable list of "suspect" class
     pairs / groups (defaults: chair/table, and the person subclasses
     boy/girl/man/woman), showing confusion counts for teacher vs plain vs
     KD student, since the aggregate superclass-confusability correlation
     may be averaging away pair-specific effects.

NOTE ON SCOPE: raw_arrays.npz stores argmax predictions, not raw logits, so
ECE cannot be bootstrapped here. If you want a bootstrap CI on ECE too, add
`logits_teacher=teacher_logits.numpy(), logits_plain=plain_logits.numpy(),
logits_kd=kd_logits.numpy()` to the np.savez(...) call in analyze.py's
"save everything" section and re-run analyze.py once (no retraining needed,
just re-inference from the existing checkpoints).

Usage:
    python significance_analysis.py --results_dir results/

No GPU, dataset, or model checkpoints needed -- this only reads the
raw_arrays.npz file that analyze.py already saved (predictions + labels),
so it runs comfortably on a laptop CPU.
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest

# Standard CIFAR-100 fine label names, in official dataset order (index 0-99).
# Hardcoded here instead of calling data.py's get_cifar100_loaders(), which
# would download the full ~170MB dataset just to read off these 100 strings
# -- unnecessary and a common source of local-machine download failures.
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



# ---------------------------------------------------------------------------
# 1. Per-class McNemar's test
# ---------------------------------------------------------------------------

def per_class_mcnemar(preds_plain, preds_kd, labels, num_classes, min_n=1):
    """
    For each class c, restrict to the ~100 test examples with true label c.
    Build the 2x2 agreement table between (plain correct?) and (kd correct?):
        n11 = both correct        n10 = plain correct, kd wrong
        n01 = plain wrong, kd correct   n00 = both wrong
    McNemar's test only uses the discordant pairs (n10, n01) -- it asks
    "given this class flipped, was it more likely to flip toward KD-correct
    or KD-wrong than a fair coin?" Exact binomial version (appropriate for
    small n like this) via scipy's binomtest on n10 vs n10+n01.
    """
    results = []
    for c in range(num_classes):
        mask = labels == c
        n = mask.sum()
        if n < min_n:
            continue
        correct_plain = (preds_plain[mask] == labels[mask])
        correct_kd = (preds_kd[mask] == labels[mask])

        n11 = int((correct_plain & correct_kd).sum())
        n10 = int((correct_plain & ~correct_kd).sum())   # KD regressed these
        n01 = int((~correct_plain & correct_kd).sum())   # KD fixed these
        n00 = int((~correct_plain & ~correct_kd).sum())

        n_discordant = n10 + n01
        if n_discordant == 0:
            p_value = 1.0
        else:
            # two-sided exact binomial test: is n10 unusually far from
            # n_discordant/2?
            p_value = binomtest(n10, n_discordant, p=0.5, alternative="two-sided").pvalue

        delta = (correct_kd.float().mean() - correct_plain.float().mean()).item() \
            if hasattr(correct_plain, "float") else (correct_kd.mean() - correct_plain.mean())

        results.append({
            "class_idx": c,
            "n_test_examples": int(n),
            "delta": float(delta),
            "n_kd_regressed": n10,
            "n_kd_improved": n01,
            "n_discordant": n_discordant,
            "mcnemar_p": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
        })
    return results


# ---------------------------------------------------------------------------
# 2. Bootstrap CIs on headline scalar metrics
# ---------------------------------------------------------------------------

def bootstrap_headline_metrics(preds_plain, preds_kd, labels, num_classes,
                                n_boot=2000, seed=0, delta_thresh=0.01):
    """
    Resample test examples with replacement (stratified within each true
    class, so every bootstrap sample still has ~100 examples per class --
    matches the original evaluation protocol instead of introducing a new
    source of imbalance) and recompute the headline scalars each time.
    Returns point estimate + 95% percentile CI for each metric.
    """
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    boot_overall_delta = np.empty(n_boot)
    boot_mean_per_class_delta = np.empty(n_boot)
    boot_std_per_class_delta = np.empty(n_boot)
    boot_n_hurt = np.empty(n_boot, dtype=int)
    boot_n_helped = np.empty(n_boot, dtype=int)

    for b in range(n_boot):
        per_class_delta = np.empty(num_classes)
        n_correct_plain = 0
        n_correct_kd = 0
        n_total = 0
        for c in range(num_classes):
            idx = class_indices[c]
            if len(idx) == 0:
                per_class_delta[c] = np.nan
                continue
            sampled = rng.choice(idx, size=len(idx), replace=True)
            cp = (preds_plain[sampled] == labels[sampled]).mean()
            ck = (preds_kd[sampled] == labels[sampled]).mean()
            per_class_delta[c] = ck - cp
            n_correct_plain += (preds_plain[sampled] == labels[sampled]).sum()
            n_correct_kd += (preds_kd[sampled] == labels[sampled]).sum()
            n_total += len(sampled)

        boot_overall_delta[b] = (n_correct_kd - n_correct_plain) / n_total
        boot_mean_per_class_delta[b] = np.nanmean(per_class_delta)
        boot_std_per_class_delta[b] = np.nanstd(per_class_delta)
        boot_n_hurt[b] = int((per_class_delta < -delta_thresh).sum())
        boot_n_helped[b] = int((per_class_delta > delta_thresh).sum())

    def ci(arr):
        return {
            "point_estimate": float(np.mean(arr)),
            "ci_lower_2.5": float(np.percentile(arr, 2.5)),
            "ci_upper_97.5": float(np.percentile(arr, 97.5)),
        }

    return {
        "overall_acc_delta_kd_minus_plain": ci(boot_overall_delta),
        "mean_per_class_delta": ci(boot_mean_per_class_delta),
        "std_per_class_delta": ci(boot_std_per_class_delta),
        "n_classes_hurt_by_kd": ci(boot_n_hurt),
        "n_classes_helped_by_kd": ci(boot_n_helped),
        "n_bootstrap_resamples": n_boot,
    }


def expected_calibration_error(logits, preds, labels, n_bins=15):
    """Same definition as analyze.py's ECE (confidence-vs-accuracy gap,
    averaged across bins, weighted by bin occupancy), reimplemented here in
    numpy so this script has no torch dependency."""
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            ece += abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop
    return ece


def bootstrap_ece_delta(logits_plain, logits_kd, preds_plain, preds_kd, labels,
                         num_classes, n_boot=2000, seed=0):
    """
    Bootstrap CI on the calibration trade-off: ECE(kd_student) -
    ECE(plain_student). Stratified resampling within each true class, same
    protocol as bootstrap_headline_metrics, so it's directly comparable.
    """
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    all_idx_pool = np.concatenate(class_indices)

    boot_ece_plain = np.empty(n_boot)
    boot_ece_kd = np.empty(n_boot)
    boot_ece_delta = np.empty(n_boot)

    for b in range(n_boot):
        sampled_chunks = []
        for c in range(num_classes):
            idx = class_indices[c]
            if len(idx) == 0:
                continue
            sampled_chunks.append(rng.choice(idx, size=len(idx), replace=True))
        sampled = np.concatenate(sampled_chunks)

        ece_p = expected_calibration_error(logits_plain[sampled], preds_plain[sampled], labels[sampled])
        ece_k = expected_calibration_error(logits_kd[sampled], preds_kd[sampled], labels[sampled])
        boot_ece_plain[b] = ece_p
        boot_ece_kd[b] = ece_k
        boot_ece_delta[b] = ece_k - ece_p

    def ci(arr):
        return {
            "point_estimate": float(np.mean(arr)),
            "ci_lower_2.5": float(np.percentile(arr, 2.5)),
            "ci_upper_97.5": float(np.percentile(arr, 97.5)),
        }

    return {
        "ece_plain_student": ci(boot_ece_plain),
        "ece_kd_student": ci(boot_ece_kd),
        "ece_delta_kd_minus_plain": ci(boot_ece_delta),
        "n_bootstrap_resamples": n_boot,
    }


# ---------------------------------------------------------------------------
# 3. Pairwise / group confusion analysis for suspect classes
# ---------------------------------------------------------------------------

def confusion_submatrix(preds, labels, class_idxs):
    """Row-normalized confusion matrix restricted to `class_idxs`, i.e.
    P(predicted = j | true = i) for i, j in class_idxs. Off-diagonal cells
    show confusion FROM row-class TO column-class."""
    k = len(class_idxs)
    mat = np.zeros((k, k), dtype=int)
    for i_row, ci in enumerate(class_idxs):
        mask = labels == ci
        n = mask.sum()
        if n == 0:
            continue
        for i_col, cj in enumerate(class_idxs):
            mat[i_row, i_col] = int((preds[mask] == cj).sum())
    return mat


def suspect_pair_report(preds_teacher, preds_plain, preds_kd, labels,
                         class_names, groups):
    """
    groups: list of lists of class names, e.g.
        [["chair", "table"], ["boy", "girl", "man", "woman"]]
    For each group, report the confusion submatrix for teacher / plain / kd,
    plus a simple "within-group confusion rate" summary scalar per model:
    fraction of examples whose true class is in the group that were
    predicted as a DIFFERENT class within the same group.
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    report = []
    for group in groups:
        missing = [n for n in group if n not in name_to_idx]
        if missing:
            report.append({"group": group, "error": f"class names not found: {missing}"})
            continue
        idxs = [name_to_idx[n] for n in group]

        entry = {"group": group, "class_indices": idxs}
        for model_name, preds in [("teacher", preds_teacher),
                                   ("plain_student", preds_plain),
                                   ("kd_student", preds_kd)]:
            mat = confusion_submatrix(preds, labels, idxs)
            n_per_class = mat.sum(axis=1)
            within_group_wrong = mat.sum() - np.trace(mat)
            total = n_per_class.sum()
            entry[model_name] = {
                "confusion_matrix": mat.tolist(),
                "row_order_(true_class)": group,
                "col_order_(predicted_class)": group,
                "within_group_confusion_rate": float(within_group_wrong / total) if total else None,
                "n_examples": int(total),
            }
        report.append(entry)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--suspect_groups", default=None,
        help="JSON string overriding default suspect groups, e.g. "
             '\'[["chair","table"],["boy","girl","man","woman"]]\''
    )
    args = p.parse_args()

    arr_path = os.path.join(args.results_dir, "raw_arrays.npz")
    data = np.load(arr_path)
    labels = data["labels"]
    preds_teacher = data["preds_teacher"]
    preds_plain = data["preds_plain"]
    preds_kd = data["preds_kd"]
    num_classes = int(labels.max()) + 1
    class_names = CIFAR100_FINE_LABELS

    out = {}

    # 1. per-class McNemar
    mcnemar_results = per_class_mcnemar(preds_plain, preds_kd, labels, num_classes)
    n_sig = sum(r["significant_at_0.05"] for r in mcnemar_results)
    n_sig_hurt = sum(r["significant_at_0.05"] and r["delta"] < 0 for r in mcnemar_results)
    n_sig_helped = sum(r["significant_at_0.05"] and r["delta"] > 0 for r in mcnemar_results)
    out["mcnemar_per_class"] = mcnemar_results
    out["mcnemar_summary"] = {
        "n_classes_tested": len(mcnemar_results),
        "n_significant_at_0.05": n_sig,
        "n_significant_hurt_by_kd": n_sig_hurt,
        "n_significant_helped_by_kd": n_sig_helped,
        "note": "No multiple-comparisons correction applied above; see "
                "'n_significant_after_bh_correction' for a Benjamini-Hochberg "
                "corrected count across the 100 class-level tests.",
    }

    # Benjamini-Hochberg correction across the 100 tests (running 100 tests
    # at alpha=0.05 uncorrected will produce ~5 false positives by chance)
    pvals = np.array([r["mcnemar_p"] for r in mcnemar_results])
    order = np.argsort(pvals)
    m = len(pvals)
    bh_thresh = (np.arange(1, m + 1) / m) * 0.05
    sorted_p = pvals[order]
    below = sorted_p <= bh_thresh
    n_bh_sig = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    out["mcnemar_summary"]["n_significant_after_bh_correction"] = n_bh_sig

    # 2. bootstrap CIs
    print("Running bootstrap (this loops per-class per-resample; may take "
          "a minute for n_boot=%d)..." % args.n_boot)
    out["bootstrap_ci"] = bootstrap_headline_metrics(
        preds_plain, preds_kd, labels, num_classes,
        n_boot=args.n_boot, seed=args.seed,
    )

    # 3. suspect pair confusion analysis
    default_groups = [["chair", "table"], ["boy", "girl", "man", "woman"]]
    groups = json.loads(args.suspect_groups) if args.suspect_groups else default_groups
    out["suspect_group_confusion"] = suspect_pair_report(
        preds_teacher, preds_plain, preds_kd, labels, class_names, groups
    )

    # 4. ECE bootstrap CI -- only if analyze.py was re-run with the logits
    # patch (adds logits_plain / logits_kd to raw_arrays.npz). Skips
    # gracefully with a clear message otherwise, rather than failing.
    if "logits_plain" in data.files and "logits_kd" in data.files:
        print("Logits found -- running ECE bootstrap "
              "(n_boot=%d)..." % args.n_boot)
        out["ece_bootstrap_ci"] = bootstrap_ece_delta(
            data["logits_plain"], data["logits_kd"],
            preds_plain, preds_kd, labels, num_classes,
            n_boot=args.n_boot, seed=args.seed,
        )
    else:
        out["ece_bootstrap_ci"] = None
        print("\nNOTE: raw_arrays.npz has no saved logits, so the ECE "
              "calibration-tradeoff CI was skipped. Re-run analyze.py with "
              "the logits-saving patch, then re-run this script, to get it.")

    out_path = os.path.join(args.results_dir, "significance_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print("\n=== McNemar per-class significance ===")
    print(json.dumps(out["mcnemar_summary"], indent=2))
    print("\n=== Bootstrap 95% CIs ===")
    print(json.dumps(out["bootstrap_ci"], indent=2))
    if out["ece_bootstrap_ci"] is not None:
        print("\n=== ECE calibration-tradeoff bootstrap 95% CI ===")
        print(json.dumps(out["ece_bootstrap_ci"], indent=2))
    print("\n=== Suspect group within-group confusion rates ===")
    for g in out["suspect_group_confusion"]:
        if "error" in g:
            print(g)
            continue
        print(f"\nGroup: {g['group']}")
        for model_name in ["teacher", "plain_student", "kd_student"]:
            print(f"  {model_name}: within-group confusion rate = "
                  f"{g[model_name]['within_group_confusion_rate']:.3f} "
                  f"(n={g[model_name]['n_examples']})")

    print(f"\nFull detail written to {out_path}")


if __name__ == "__main__":
    main()