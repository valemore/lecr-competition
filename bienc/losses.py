# Loss functions and score functions for Bi-encoder
import torch
from torch.nn import functional as F


class BidirectionalMarginLoss:
    def __init__(self, device: torch.device, margin: float = 0.0):
        """
        Loss function for multiple negatives ranking loss with optional margin.
        Bi-directional means that we compute cross entropy loss both with respect to topic (given a topic embedding, pick the right content),
        and with respect to content (given a content embedding, pick the right topic).
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

    def to(self, device):
        self.device = device
        return self


class UnidirectionalMarginLoss:
    def __init__(self, device: torch.device, margin: float = 0.0):
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

    def to(self, device):
        self.device = device
        return self


def dot_score(topic_emb, content_emb):
    """Computes matrix of dot product scores between CONTENT_EMB and TOPIC_EMB."""
    return content_emb.mm(topic_emb.t())


def cos_sim(topic_emb, content_emb):
    """Computes matrix of scaled cosine similarities between CONTENT_EMB and TOPIC_EMB. Scaling factor is 20."""
    # Scaling factor is chosen to make things easier for the subsequent softmax
    return F.cosine_similarity(topic_emb.t().unsqueeze(0), content_emb.unsqueeze(-1), dim=1) * 20 # TODO: ???
