import torch

def binary_accuracy(logits, targets):
    """
        Computes the accuracy for binary classification based on model output logits.

        Args:
            logits (torch.Tensor): Raw output from the model before activation (shape: [batch_size]).
            targets (torch.Tensor): Ground truth binary labels (0 or 1), same shape as logits.

        Returns:
            torch.Tensor: The accuracy as a fraction (number of correct predictions / total samples).
    """
    # Apply sigmoid to logits to get probabilities between 0 and 1
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions: threshold at 0.5
    preds = (probs > 0.5).float()

    #  Count how many predictions match the targets
    correct = (preds == targets).float().sum()

    # Calculate accuracy as correct predictions divided by total number of samples
    acc = correct / targets.numel()

    return acc


def precision_recall_f1(logits, targets):
    """
        Computes precision, recall, and F1-score for binary classification predictions.

        Args:
            logits (torch.Tensor): Raw model outputs before activation (shape: [batch_size]).
            targets (torch.Tensor): Ground truth binary labels (0 or 1), same shape as logits.

        Returns:
            tuple: (precision, recall, f1) all as floats.
                - precision: ratio of true positives to all positive predictions.
                - recall: ratio of true positives to all actual positives.
                - f1: harmonic mean of precision and recall.
    """
    # Convert logits to binary predictions by sigmoid thresholding
    preds = (torch.sigmoid(logits) > 0.5).float()

    # True positives: predictions = 1 and targets = 1
    tp = ((preds ==1) & (targets == 1)).sum().item()

    # False positives: predictions = 1 but targets = 0
    fp = ((preds == 1) & (targets ==0)).sum().item()

    # False negatives: predictions = 0 but targets = 1
    fn = ((preds == 0) & (targets == 1)).sum().item()

    # To avoid division by zero, add a small epsilon (1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fp + 1e-6)

    # Harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1