import torch
import numpy as np
from data import get_cifar100_loaders
from models import get_student, get_teacher
from diagnostics import collect_predictions, get_per_class_accuracy, expected_calibration_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, test_loader, class_names = get_cifar100_loaders()

# 1. Load your models
# (Assuming you have run the training loops and saved the state_dicts)
teacher = get_teacher().to(device)
student_base = get_student().to(device)
student_kd = get_student().to(device)

# 2. Extract Data
t_preds, t_probs, targets = collect_predictions(teacher, test_loader, device)
base_preds, base_probs, _ = collect_predictions(student_base, test_loader, device)
kd_preds, kd_probs, _     = collect_predictions(student_kd, test_loader, device)

# 3. Calculate Per-Class Accuracies
t_accs, _    = get_per_class_accuracy(targets, t_preds)
base_accs, _ = get_per_class_accuracy(targets, base_preds)
kd_accs, _   = get_per_class_accuracy(targets, kd_preds)

# 4. Generate Headline Deltas
kd_delta = kd_accs - base_accs  # Positive means KD helped; Negative means KD hurt
teacher_gap = t_accs - base_accs

# 5. Calibration
print(f"Teacher ECE: {expected_calibration_error(t_probs, targets):.4f}")
print(f"Baseline Student ECE: {expected_calibration_error(base_probs, targets):.4f}")
print(f"KD Student ECE: {expected_calibration_error(kd_probs, targets):.4f}")

# 6. Find 'Flipped' Examples for deep-dive analysis
# Examples where teacher was right, baseline student was right, but KD ruined it.
hurt_by_kd = (t_preds == targets) & (base_preds == targets) & (kd_preds != targets)
hurt_indices = np.where(hurt_by_kd)[0]
print(f"Number of pristine examples broken explicitly by KD: {len(hurt_indices)}")