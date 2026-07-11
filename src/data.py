import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(batch_size=128, num_workers=2):
    # ImageNet normalization since the teacher is pretrained
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes