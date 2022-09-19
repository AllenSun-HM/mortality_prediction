import sklearn
import torch
import torch.nn as nn
from sklearn.utils import class_weight
from torch.nn import functional as F
import numpy as np

def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.∂
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        class_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(np.ravel(target.cpu(),order='C')), y=np.ravel(target.cpu(),order='C'))
        self.weight = torch.tensor(class_w, dtype=torch.float)
        if self.weight.shape == torch.Size([1]): # input shape [1] weight tensor would cause error (this happens when all labels in the batch are the same)
            return F.cross_entropy(inp, target.long().squeeze(),
                                   ignore_index=self.ignore_index, reduction=self.reduction)
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                                   ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)

def auroc(probs, targets):
    false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
    return sklearn.metrics.auc(false_pos_rate, true_pos_rate)

def auprc(probs, targets):
    prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
    return sklearn.metrics.auc(rec, prec)
