import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):    # CB-FocalLoss
    def __init__(self, beta=0.9999, gamma=1.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        labels = labels.view(-1)
        logits = logits.view(-1, logits.size(-1))

        num_labels = torch.bincount(labels, minlength=logits.size(1)).float()
        effective_num = 1.0 - torch.pow(self.beta, num_labels)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / torch.sum(weights) * logits.size(1)  # normalize

        weights = weights.to(logits.device)

        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        logits: [B, C] or [B] (before softmax or sigmoid)
        targets: [B] or [B, C] (long or one-hot)
        """

        # 判斷是 binary classification 還是 multi-class
        if logits.dim() == 1 or logits.size(1) == 1:
            # 二分類：使用 sigmoid
            probs = torch.sigmoid(logits.view(-1))
            targets = targets.float().view(-1)
            tp = (probs * targets).sum()
            fp = (probs * (1 - targets)).sum()
            fn = ((1 - probs) * targets).sum()
        else:
            # 多分類：使用 softmax
            probs = torch.softmax(logits, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
            probs = probs.view(-1, logits.size(1))
            targets_one_hot = targets_one_hot.view(-1, logits.size(1))

            tp = (probs * targets_one_hot).sum(dim=0)
            fp = (probs * (1 - targets_one_hot)).sum(dim=0)
            fn = ((1 - probs) * targets_one_hot).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        return 1 - f1.mean()

# class FocalLoss(nn.Module):

#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         probas = torch.softmax(inputs, dim=1)
#         pt = probas.gather(1, targets.unsqueeze(1)).squeeze(1)  # 選擇正確類別的概率 
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss