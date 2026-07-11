import torch
import numpy as np
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Helper to gather all predictions, probabilities, and true labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        torch_preds = torch.argmax(probs, dim=1)
        
        all_preds.append(torch_preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
        
    return (np.concatenate(all_preds), 
            np.concatenate(all_probs), 
            np.concatenate(all_targets))

def get_per_class_accuracy(targets, preds, num_classes=100):
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    # Accuracy per class = diagonal / row sums
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    return class_acc, cm

def expected_calibration_error(probs, targets, n_bins=15):
    """Calculates ECE to see if KD improves/degrades confidence alignment."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    confidencies = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidencies > bin_lower) & (confidencies <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidencies[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece