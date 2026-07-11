import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        if teacher is not None:
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            loss = criterion(outputs, teacher_outputs, targets)
        else:
            loss = nn.functional.cross_entropy(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return running_loss / total, 100.0 * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total