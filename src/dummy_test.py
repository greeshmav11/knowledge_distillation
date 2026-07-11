# dummy_test.py
import torch
from data import get_cifar100_loaders
from models import get_student, get_teacher
from loss import KDLoss
from data.train import train_epoch

print("Testing environment and data loading...")
# Use a tiny batch size for a quick local test
train_loader, _, _ = get_cifar100_loaders(batch_size=4)

device = torch.device("cpu") # Test on CPU locally to save setup time
student = get_student().to(device)
teacher = get_teacher().to(device)
criterion = KDLoss(temperature=4.0, alpha=0.5)
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

print("Running a single dummy training iteration...")
# Pull exactly one batch to see if the forward and backward passes work
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

# Forward passes
s_out = student(inputs)
t_out = teacher(inputs)

# Loss check
loss = criterion(s_out, t_out, targets)
loss.backward()
optimizer.step()

print("Success! No shape mismatches or syntax errors. Code is ready for the cluster.")