import torch

def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    return torch.sum((indices==targets).int(),dim=1).cpu().tolist()

def get_mrr(indices, targets,device):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1, 1).expand_as(indices)
    indices,value = torch.nonzero((indices==targets).int(),as_tuple=True)
    zero = torch.zeros(targets.shape[0], dtype=torch.int64).to(device)
    zero[indices] = (value + 2)
    zero = torch.reciprocal(torch.log2(zero.float()))
    mask = torch.isinf(zero)
    zero = torch.masked_fill(zero,mask,value=torch.tensor(0.))

    return zero.cpu().tolist()


def evaluate(indices, targets, device, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """

    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets, device)
    return recall, mrr
