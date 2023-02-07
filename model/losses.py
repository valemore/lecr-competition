# Loss functions and score functions for Bi-encoder
import torch
from torch.nn import functional as F


class BidirectionalMarginLoss:
    def __init__(self, device, margin=0.0):
        """
        Loss function for multiple negatives ranking loss with optional margin.
        Bi-directional means that we compute cross entropy loss both with respect to NIA entity (given a query embedding, pick the right entity),
        and with respect to query (given a NIA embedding, pick the right query).
        :param device: torch device to use
        :param margin: margin to substract from true class scores
        """
        self.device = device
        self.margin = margin

    def __call__(self, scores, mask):
        """
        Computes loss and gradients.
        :param scores: tensor of scores of shape (batch_size, batch_size)
        :param mask: boolean tensor of shape (batch_size, batch_size) indicating where we zero out the loss
        :return:
        """
        scores -= torch.diagflat(torch.full((scores.shape[0],), self.margin, device=self.device))
        bs = scores.shape[0]
        target = torch.arange(bs, device=self.device)
        # Zeroing out loss when we have the same entity more than once in a batch
        scores[mask] = float('-inf')
        loss = F.cross_entropy(scores, target, reduction="mean") + F.cross_entropy(scores.t(), target, reduction="mean")
        return loss


class UnidirectionalMarginLoss:
    def __init__(self, device, margin=0.0):
        """
        Loss function for multiple negatives ranking loss with optional margin.
        :param device: torch device to use
        :param margin: margin to substract from true class scores
        """
        self.device = device
        self.margin = margin

    def __call__(self, scores, mask):
        """
        Computes loss and gradients.
        :param scores: tensor of scores of shape (batch_size, batch_size)
        :param mask: boolean tensor of shape (batch_size, batch_size) indicating where we zero out the loss
        :return:
        """
        scores -= torch.diagflat(torch.full((scores.shape[0],), self.margin, device=self.device))
        bs = scores.shape[0]
        target = torch.arange(bs, device=self.device)
        # Zeroing out loss when we have the same entity more than once in a batch
        scores[mask] = float('-inf')
        loss = F.cross_entropy(scores, target, reduction="mean")
        return loss


def dot_score(query_emb, ent_emb):
    """Computes matrix of dot product scores between QUERY_EMB and ENT_EMB."""
    return query_emb.mm(ent_emb.t())


def cos_sim(query_emb, ent_emb):
    """Computes matrix of scaled cosine similarities between QUERY_EMB and ENT_EMB. Scaling factor is 20."""
    # Scaling factor is chosen to make things easier for the subsequent softmax
    return F.cosine_similarity(query_emb.unsqueeze(-1), ent_emb.t().unsqueeze(0), dim=1) * 20
