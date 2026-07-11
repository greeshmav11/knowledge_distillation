"""
Knowledge distillation losses.

- kd_loss: the standard Hinton et al. (2015) loss:
    L = alpha * CE(student_logits, labels)
        + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))

- class_balanced_kd_loss: an optional mitigation variant that reweights the
  per-example loss (both the CE and the KD terms) by an inverse-effective-
  number class weight (Cui et al., 2019, "Class-Balanced Loss Based on
  Effective Number of Samples"), so that the loss doesn't uniformly favor
  majority classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """
    student_logits, teacher_logits: (B, C) raw logits
    labels: (B,) long tensor of ground-truth class indices
    T: distillation temperature
    alpha: weight on the hard-label CE loss; (1 - alpha) on the soft KD loss
    """
    ce = F.cross_entropy(student_logits, labels)

    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    # batchmean reduction + T^2 scaling as in the original KD paper
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

    loss = alpha * ce + (1.0 - alpha) * kd
    return loss, {"ce": ce.item(), "kd": kd.item()}


def effective_number_weights(class_counts, num_classes, beta=0.999):
    """
    class_counts: dict or array-like of length num_classes with training
                  example counts per class (e.g. meta['class_counts'] from
                  get_cifar100_lt_loaders).
    Returns a (num_classes,) tensor of per-class weights, normalized so
    weights sum to num_classes (keeps the overall loss scale comparable
    to the unweighted case).
    """
    if isinstance(class_counts, dict):
        counts = torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float)
    else:
        counts = torch.tensor(class_counts, dtype=torch.float)

    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return weights


def class_balanced_kd_loss(student_logits, teacher_logits, labels, class_weights,
                            T=4.0, alpha=0.5):
    """
    Same as kd_loss but reweights both the CE and KD terms per-example by
    the class weight of that example's ground-truth label. class_weights is
    a (num_classes,) tensor, e.g. from effective_number_weights(...).
    """
    device = student_logits.device
    class_weights = class_weights.to(device)
    per_example_w = class_weights[labels]  # (B,)

    ce_per_example = F.cross_entropy(student_logits, labels, reduction="none")
    ce = (ce_per_example * per_example_w).mean()

    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kd_per_example = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    kd = (kd_per_example * per_example_w).mean() * (T ** 2)

    loss = alpha * ce + (1.0 - alpha) * kd
    return loss, {"ce": ce.item(), "kd": kd.item()}


if __name__ == "__main__":
    torch.manual_seed(0)
    s = torch.randn(8, 100)
    t = torch.randn(8, 100)
    y = torch.randint(0, 100, (8,))
    loss, parts = kd_loss(s, t, y)
    print("kd_loss:", loss.item(), parts)

    counts = {i: 500 for i in range(100)}
    counts[0] = 5  # simulate a rare class
    w = effective_number_weights(counts, 100)
    print("weight of rare class 0:", w[0].item(), "vs common class 1:", w[1].item())
    loss_cb, parts_cb = class_balanced_kd_loss(s, t, y, w)
    print("class_balanced_kd_loss:", loss_cb.item(), parts_cb)
