"""
Training entry point. Handles four run modes:

  teacher          -- fine-tune ImageNet-pretrained teacher on CIFAR-100
  student_plain    -- train student from scratch/pretrained, no distillation
  student_kd       -- train student with standard KD loss against a trained teacher
  student_kd_cb    -- train student with class-balanced KD loss (mitigation variant,
                       requires CIFAR-100-LT so class_counts are meaningful)

Examples:
  python train.py --mode teacher --arch resnet50 --epochs 30 \
      --save ckpts/teacher.pt

  python train.py --mode student_plain --student resnet18 --epochs 40 \
      --save ckpts/student_plain.pt

  python train.py --mode student_kd --student resnet18 --epochs 40 \
      --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
      --T 4.0 --alpha 0.5 --save ckpts/student_kd.pt

  python train.py --mode student_kd_cb --student resnet18 --epochs 40 \
      --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
      --lt --imb_factor 0.01 --T 4.0 --alpha 0.5 --save ckpts/student_kd_cb.pt

Checkpoints saved are plain state_dicts, easily loaded by analyze.py.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_cifar100_loaders, get_cifar100_lt_loaders
from models import build_teacher, build_student
from losses import kd_loss, class_balanced_kd_loss, effective_number_weights


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one_epoch_plain(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_one_epoch_kd(student, teacher, loader, optimizer, device, T, alpha,
                        class_weights=None):
    student.train()
    teacher.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            teacher_logits = teacher(x)
        optimizer.zero_grad()
        student_logits = student(x)
        if class_weights is not None:
            loss, _ = class_balanced_kd_loss(student_logits, teacher_logits, y,
                                              class_weights, T=T, alpha=alpha)
        else:
            loss, _ = kd_loss(student_logits, teacher_logits, y, T=T, alpha=alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                    choices=["teacher", "student_plain", "student_kd", "student_kd_cb"])
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet34"],
                    help="teacher architecture (only used when --mode teacher)")
    p.add_argument("--student", default="resnet18", choices=["resnet18", "small_cnn"])
    p.add_argument("--teacher_ckpt", default=None,
                    help="path to a trained teacher checkpoint (required for KD modes)")
    p.add_argument("--teacher_arch", default="resnet50", choices=["resnet50", "resnet34"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lt", action="store_true", help="use CIFAR-100-LT (long-tailed) train set")
    p.add_argument("--imb_factor", type=float, default=0.01)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--save", required=True)
    p.add_argument("--pretrained", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.lt:
        train_loader, test_loader, meta = get_cifar100_lt_loaders(
            root=args.data_root, batch_size=args.batch_size, imb_factor=args.imb_factor
        )
    else:
        train_loader, test_loader, meta = get_cifar100_loaders(
            root=args.data_root, batch_size=args.batch_size
        )

    num_classes = meta["num_classes"]

    # -------- build model(s) --------
    if args.mode == "teacher":
        model = build_teacher(arch=args.arch, num_classes=num_classes,
                               pretrained=args.pretrained).to(device)
    else:
        model = build_student(kind=args.student, num_classes=num_classes,
                               pretrained=args.pretrained).to(device)

    teacher = None
    class_weights = None
    if args.mode in ("student_kd", "student_kd_cb"):
        assert args.teacher_ckpt is not None, "must provide --teacher_ckpt for KD modes"
        teacher = build_teacher(arch=args.teacher_arch, num_classes=num_classes,
                                 pretrained=False).to(device)
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)

    if args.mode == "student_kd_cb":
        assert args.lt, "--lt must be set (class_counts needed) for student_kd_cb"
        class_weights = effective_number_weights(meta["class_counts"], num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        if args.mode in ("student_kd", "student_kd_cb"):
            train_loss = train_one_epoch_kd(model, teacher, train_loader, optimizer,
                                             device, T=args.T, alpha=args.alpha,
                                             class_weights=class_weights)
        else:
            train_loss = train_one_epoch_plain(model, train_loader, optimizer, device)
        scheduler.step()

        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        dt = time.time() - t0
        print(f"[{args.mode}] epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"best={best_acc:.4f}  ({dt:.1f}s)")

    torch.save(model.state_dict(), args.save)
    print(f"saved final checkpoint to {args.save} (final test_acc={test_acc:.4f}, best={best_acc:.4f})")


if __name__ == "__main__":
    main()
