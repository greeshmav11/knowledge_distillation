import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, class_weights=None):
        super(KDLoss, self).__init__()
        self.T = temperature
        self.alpha = alpha
        self.class_weights = class_weights # Reserved for your mitigation phase
        
    def forward(self, student_logits, teacher_logits, targets):
        # Standard Hard Label Loss
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(student_logits, targets, weight=self.class_weights)
        else:
            ce_loss = F.cross_entropy(student_logits, targets)
            
        # Soft Label Loss (KL Divergence)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        
        # kl_div expects log_target=False by default in recent PyTorch, reduction='batchmean' is mathematically correct
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        
        # Combined Loss
        loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        return loss