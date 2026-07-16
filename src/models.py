"""
Teacher and student model definitions.

Both use torchvision ResNets pretrained on ImageNet, but adapted for
32x32 CIFAR input: the 7x7/stride-2 stem conv is replaced with a
3x3/stride-1 conv and the initial maxpool is removed (standard practice
for CIFAR + ImageNet-pretrained backbones; without this, too much
spatial resolution is lost before the first residual block).

Teacher: ResNet-50 (or ResNet-34), ImageNet-pretrained, fine-tuned on CIFAR-100.
Student: ResNet-18, ImageNet-pretrained (still much smaller/faster than teacher)
         OR trained from scratch, OR a small custom CNN if you want a bigger
         capacity gap between teacher and student (recommended for a clearer
         distillation effect).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def _adapt_stem_for_cifar(model):
    """Replace the ImageNet stem (7x7 conv, stride 2 + maxpool) with a
    CIFAR-friendly stem (3x3 conv, stride 1, no maxpool)."""
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3,
                             stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_teacher(arch="resnet50", num_classes=100, pretrained=True):
    assert arch in ("resnet50", "resnet34")
    ctor = tvm.resnet50 if arch == "resnet50" else tvm.resnet34
    model = ctor(pretrained=pretrained)
    model = _adapt_stem_for_cifar(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_student_resnet18(num_classes=100, pretrained=True):
    model = tvm.resnet18(pretrained=pretrained)
    model = _adapt_stem_for_cifar(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SmallCNN(nn.Module):
    """A deliberately low-capacity CNN student, useful if you want a bigger
    teacher-student capacity gap than ResNet50->ResNet18 gives you (a bigger
    gap tends to make distillation effects, and any disparate impact, more
    visible within a 2-week timeline)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_student(kind="resnet18", num_classes=100, pretrained=True):
    if kind == "resnet18":
        return build_student_resnet18(num_classes=num_classes, pretrained=pretrained)
    elif kind == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    else:
        raise ValueError(f"unknown student kind: {kind}")


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    t = build_teacher("resnet50", pretrained=False)
    s = build_student("resnet18", pretrained=False)
    sc = build_student("small_cnn")
    print("teacher out:", t(x).shape)
    print("student resnet18 out:", s(x).shape)
    print("student small_cnn out:", sc(x).shape)
    n_t = sum(p.numel() for p in t.parameters())
    n_s = sum(p.numel() for p in s.parameters())
    n_sc = sum(p.numel() for p in sc.parameters())
    print(f"params: teacher={n_t/1e6:.1f}M  resnet18={n_s/1e6:.1f}M  small_cnn={n_sc/1e6:.2f}M")
