import sklearn
import torch
import sklearn.metrics

def cross_entropy_loss(outputs, targets, weight=None):
    criterion = torch.nn.functional.cross_entropy
    return criterion(outputs, targets.long().squeeze(), weight=torch.tensor(weight, dtype=torch.float))

def accuracy(predictions, targets):
    return torch.sum(predictions == targets).item() / len(targets)

def auroc(probs, targets):
    false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
    return sklearn.metrics.auc(false_pos_rate, true_pos_rate)

def auprc(probs, targets):
    prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
    return sklearn.metrics.auc(rec, prec)

def confusion_matrix(predictions, targets):
    sklearn.metrics.confusion_matrix(targets, predictions)