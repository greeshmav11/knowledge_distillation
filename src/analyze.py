"""
Diagnostic analysis for the distillation study. Loads teacher, plain
student, and KD student checkpoints, runs them on the CIFAR-100 test set,
and produces:

  1. Per-class accuracy for all three models + accuracy DELTA
     (distilled_student - plain_student), the headline plot.
  2. The list of test examples that flip correct -> incorrect when going
     from plain student to distilled student (and vice versa).
  3. Correlation between the per-class accuracy delta and:
       (a) class frequency (if you trained on CIFAR-100-LT), and
       (b) a confusability score derived from the 20 CIFAR-100 superclasses
           (mean teacher softmax probability mass placed on OTHER classes
           within the same superclass -- a proxy for how visually/semantically
           confusable a class is with its siblings).
  4. Expected Calibration Error (ECE) + reliability diagrams for teacher,
     plain student, and distilled student.

Usage:
    python analyze.py \
        --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
        --student_plain_ckpt ckpts/student_plain.pt \
        --student_kd_ckpt ckpts/student_kd.pt \
        --student resnet18 \
        [--lt --imb_factor 0.01]   # only if you trained on CIFAR-100-LT and
                                    # want the frequency correlation
        --out_dir results/
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from data import get_cifar100_loaders, get_cifar100_lt_loaders, get_fine_to_coarse_mapping
from models import build_teacher, build_student


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_all_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def per_class_accuracy(logits, labels, num_classes):
    preds = logits.argmax(dim=1)
    acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            acc[c] = float("nan")
        else:
            acc[c] = (preds[mask] == labels[mask]).float().mean().item()
    return acc, preds


def find_flips(preds_a, preds_b, labels):
    """Indices where model A was correct and model B is wrong (a-> b regression),
    and the reverse (b fixes what a got wrong)."""
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    regressed = ((correct_a) & (~correct_b)).nonzero(as_tuple=True)[0]
    improved = ((~correct_a) & (correct_b)).nonzero(as_tuple=True)[0]
    return regressed, improved


# ---------------------------------------------------------------------------
# Calibration / ECE
# ---------------------------------------------------------------------------

def expected_calibration_error(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_data.append((lo.item(), hi.item(), acc_in_bin.item(),
                              conf_in_bin.item(), prop_in_bin.item()))
        else:
            bin_data.append((lo.item(), hi.item(), None, None, 0.0))
    return ece.item(), bin_data


def plot_reliability_diagram(bin_data, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = [b for b in bin_data if b[2] is not None]
    mids = [(b[0] + b[1]) / 2 for b in bins]
    accs = [b[2] for b in bins]
    ax.bar(mids, accs, width=1.0 / len(bin_data), edgecolor="black", alpha=0.7, label="accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusability score from superclasses
# ---------------------------------------------------------------------------

def superclass_confusability(teacher_logits, labels, fine_to_coarse, num_classes=100):
    """
    For each fine class c, compute the mean probability mass the TEACHER
    places on other fine classes that share the same superclass, for
    examples truly belonging to class c. Higher = more confusable with its
    superclass siblings.
    """
    probs = F.softmax(teacher_logits, dim=1).numpy()
    fine_to_coarse = np.asarray(fine_to_coarse)
    confusability = np.zeros(num_classes)

    for c in range(num_classes):
        mask = (labels == c).numpy()
        if mask.sum() == 0:
            confusability[c] = np.nan
            continue
        siblings = np.where((fine_to_coarse == fine_to_coarse[c]) & (np.arange(num_classes) != c))[0]
        confusability[c] = probs[mask][:, siblings].sum(axis=1).mean()
    return confusability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--teacher_arch", default="resnet50", choices=["resnet50", "resnet34"])
    p.add_argument("--student_plain_ckpt", required=True)
    p.add_argument("--student_kd_ckpt", required=True)
    p.add_argument("--student", default="resnet18", choices=["resnet18", "small_cnn"])
    p.add_argument("--student_kd_cb_ckpt", default=None,
                    help="optional: mitigation variant checkpoint")
    p.add_argument("--lt", action="store_true")
    p.add_argument("--imb_factor", type=float, default=0.01)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: analysis always evaluates on the standard balanced CIFAR-100 test
    # set, regardless of whether training used CIFAR-100-LT, so per-class
    # accuracy numbers are directly comparable across classes.
    _, test_loader, meta = get_cifar100_loaders(root=args.data_root)
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    class_counts = None
    if args.lt:
        _, _, lt_meta = get_cifar100_lt_loaders(root=args.data_root, imb_factor=args.imb_factor)
        class_counts = lt_meta["class_counts"]

    fine_to_coarse, coarse_names, fine_names = get_fine_to_coarse_mapping(root=args.data_root)

    # -------- load models --------
    teacher = build_teacher(arch=args.teacher_arch, num_classes=num_classes, pretrained=False).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))

    student_plain = build_student(kind=args.student, num_classes=num_classes, pretrained=False).to(device)
    student_plain.load_state_dict(torch.load(args.student_plain_ckpt, map_location=device))

    student_kd = build_student(kind=args.student, num_classes=num_classes, pretrained=False).to(device)
    student_kd.load_state_dict(torch.load(args.student_kd_ckpt, map_location=device))

    student_kd_cb = None
    if args.student_kd_cb_ckpt:
        student_kd_cb = build_student(kind=args.student, num_classes=num_classes, pretrained=False).to(device)
        student_kd_cb.load_state_dict(torch.load(args.student_kd_cb_ckpt, map_location=device))

    # -------- run inference once, reuse everywhere --------
    teacher_logits, labels = get_all_logits(teacher, test_loader, device)
    plain_logits, _ = get_all_logits(student_plain, test_loader, device)
    kd_logits, _ = get_all_logits(student_kd, test_loader, device)
    cb_logits = None
    if student_kd_cb is not None:
        cb_logits, _ = get_all_logits(student_kd_cb, test_loader, device)

    # -------- 1. per-class accuracy + delta --------
    acc_teacher, preds_teacher = per_class_accuracy(teacher_logits, labels, num_classes)
    acc_plain, preds_plain = per_class_accuracy(plain_logits, labels, num_classes)
    acc_kd, preds_kd = per_class_accuracy(kd_logits, labels, num_classes)
    delta = acc_kd - acc_plain  # positive = distillation helped that class

    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(num_classes), delta[order], color=np.where(delta[order] >= 0, "tab:green", "tab:red"))
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([class_names[i] for i in order], rotation=90, fontsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("accuracy delta (KD student - plain student)")
    ax.set_title("Per-class effect of distillation")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "per_class_delta.png"), dpi=150)
    plt.close(fig)

    summary = {
        "overall_acc_teacher": float((preds_teacher == labels).float().mean()),
        "overall_acc_plain_student": float((preds_plain == labels).float().mean()),
        "overall_acc_kd_student": float((preds_kd == labels).float().mean()),
        "mean_per_class_delta": float(np.nanmean(delta)),
        "std_per_class_delta": float(np.nanstd(delta)),
        "n_classes_hurt_by_kd": int((delta < -0.01).sum()),
        "n_classes_helped_by_kd": int((delta > 0.01).sum()),
        "worst_hurt_classes": [class_names[i] for i in order[:10]],
        "most_helped_classes": [class_names[i] for i in order[-10:]],
    }

    # -------- 2. flip analysis --------
    regressed, improved = find_flips(torch.from_numpy(np.array(preds_plain)),
                                      torch.from_numpy(np.array(preds_kd)), labels)
    summary["n_examples_regressed_by_kd"] = int(len(regressed))
    summary["n_examples_improved_by_kd"] = int(len(improved))
    np.save(os.path.join(args.out_dir, "regressed_indices.npy"), regressed.numpy())
    np.save(os.path.join(args.out_dir, "improved_indices.npy"), improved.numpy())

    # class breakdown of regressions (which true classes lose examples most)
    regressed_labels = labels[regressed].numpy()
    reg_counts = np.bincount(regressed_labels, minlength=num_classes)
    top_regressed_classes = np.argsort(-reg_counts)[:10]
    summary["classes_with_most_regressions"] = [
        {"class": class_names[c], "n_regressed": int(reg_counts[c])} for c in top_regressed_classes
    ]

    # -------- 3. correlation: delta vs frequency / confusability --------
    confusability = superclass_confusability(teacher_logits, labels, fine_to_coarse, num_classes)
    valid = ~np.isnan(delta) & ~np.isnan(confusability)
    r_conf, p_conf = pearsonr(delta[valid], confusability[valid])
    rho_conf, p_conf_s = spearmanr(delta[valid], confusability[valid])
    summary["corr_delta_vs_confusability_pearson"] = {"r": float(r_conf), "p": float(p_conf)}
    summary["corr_delta_vs_confusability_spearman"] = {"rho": float(rho_conf), "p": float(p_conf_s)}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(confusability, delta, s=15)
    ax.set_xlabel("teacher confusability with superclass siblings")
    ax.set_ylabel("accuracy delta (KD - plain)")
    ax.set_title(f"Delta vs confusability (pearson r={r_conf:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "delta_vs_confusability.png"), dpi=150)
    plt.close(fig)

    if class_counts is not None:
        freq = np.array([class_counts.get(c, 0) for c in range(num_classes)])
        valid_f = ~np.isnan(delta)
        r_freq, p_freq = pearsonr(delta[valid_f], freq[valid_f])
        rho_freq, p_freq_s = spearmanr(delta[valid_f], freq[valid_f])
        summary["corr_delta_vs_frequency_pearson"] = {"r": float(r_freq), "p": float(p_freq)}
        summary["corr_delta_vs_frequency_spearman"] = {"rho": float(rho_freq), "p": float(p_freq_s)}

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(freq, delta, s=15)
        ax.set_xlabel("training examples for this class (CIFAR-100-LT)")
        ax.set_ylabel("accuracy delta (KD - plain)")
        ax.set_title(f"Delta vs class frequency (pearson r={r_freq:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "delta_vs_frequency.png"), dpi=150)
        plt.close(fig)

    # -------- 4. calibration / ECE --------
    for name, logits in [("teacher", teacher_logits), ("student_plain", plain_logits),
                          ("student_kd", kd_logits)] + \
                         ([("student_kd_cb", cb_logits)] if cb_logits is not None else []):
        ece, bin_data = expected_calibration_error(logits, labels)
        summary[f"ece_{name}"] = float(ece)
        plot_reliability_diagram(bin_data, f"Reliability diagram: {name} (ECE={ece:.4f})",
                                  os.path.join(args.out_dir, f"reliability_{name}.png"))

    # -------- save everything --------
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(os.path.join(args.out_dir, "raw_arrays.npz"),
              acc_teacher=acc_teacher, acc_plain=acc_plain, acc_kd=acc_kd,
              delta=delta, confusability=confusability,
              labels=labels.numpy(), preds_teacher=preds_teacher.numpy(),
              preds_plain=preds_plain.numpy(), preds_kd=preds_kd.numpy(),
              logits_teacher=teacher_logits.numpy(), logits_plain=plain_logits.numpy(),
              logits_kd=kd_logits.numpy())

    print(json.dumps(summary, indent=2))
    print(f"\nAll plots and summary.json written to {args.out_dir}/")


if __name__ == "__main__":
    main()