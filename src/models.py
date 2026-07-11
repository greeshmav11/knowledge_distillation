import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

def get_teacher():
    # Pretrained on ImageNet
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Adapt final fully connected layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def get_student(pretrained=False):
    # ResNet18 as student
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model