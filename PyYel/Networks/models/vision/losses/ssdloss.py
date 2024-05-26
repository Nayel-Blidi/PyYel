import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDLoss(nn.Module):
    def __init__(self, localization_weight=1.0, confidence_weight=1.0, neg_pos_ratio=3):
        super(SSDLoss, self).__init__()
        self.localization_weight = localization_weight
        self.confidence_weight = confidence_weight
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Compute the SSD MultiBox Loss.

        Args:
            predicted_locs (list): predicted locations from SSD model
            predicted_scores (list): predicted class scores from SSD model
            boxes (list): ground truth boxes
            labels (list): ground truth labels

        Returns:
            loss (tensor): total loss
        """
        # Convert lists of dictionaries to tensors
        predicted_locs_tensor = torch.cat([torch.stack([pred['boxes'] for pred in preds]) for preds in predicted_locs], dim=0)
        predicted_scores_tensor = torch.cat([torch.stack([pred['scores'] for pred in preds]) for preds in predicted_scores], dim=0)

        # Localization loss (Smooth L1)
        localization_loss = F.smooth_l1_loss(predicted_locs_tensor, boxes)

        # Confidence loss (Cross-Entropy)
        num_classes = predicted_scores_tensor.size(-1)
        with torch.no_grad():
            positive_mask = labels > 0
            num_positives = positive_mask.sum(dim=1)
            num_negatives = torch.clamp(self.neg_pos_ratio * num_positives, max=boxes.size(1) - 1)
        
        confidence_loss = F.cross_entropy(predicted_scores_tensor.view(-1, num_classes), labels.view(-1), reduction='none')
        confidence_loss = confidence_loss.view(predicted_scores_tensor.size(0), -1)
        confidence_loss[positive_mask] = 0  # Ignore positive examples
        confidence_loss, _ = confidence_loss.sort(dim=1, descending=True)
        hardness_rank = torch.arange(confidence_loss.size(1), device=confidence_loss.device).unsqueeze(0)
        hard_negatives = hardness_rank < num_negatives.unsqueeze(1)

        confidence_loss = confidence_loss[hard_negatives].sum()

        # Total loss
        loss = self.localization_weight * localization_loss + self.confidence_weight * confidence_loss
        return loss
