import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neptune.new import Run

from utils import get_learning_rate_momentum


class LitBienc(pl.LightningModule):
    def __init__(self, bienc, loss_fn, learning_rate, weigth_decay, run):
        super().__init__()
        self.bienc = bienc
        self.loss_fn = loss_fn
        self.max_lr = learning_rate
        self.weight_decay = weigth_decay
        self.run = run

    def training_step(self, batch, batch_idx):
        *model_input, topic_idxs, content_idxs = batch

        scores = self.bienc(*model_input)
        mask = torch.full_like(scores, False, dtype=torch.bool)
        mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
        mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
        mask.fill_diagonal_(False)
        loss = self.loss_fn(scores, mask)

        # Log
        self.run["train/loss"].log(loss.item(), step=batch_idx)
        lr, momentum = get_learning_rate_momentum(self.optimizers()[0]._optimizer)
        self.run["lr"].log(lr, step=batch_idx)
        if momentum:
            self.run["momentum"].log(momentum, step=batch_idx)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        *model_input, topic_idxs, content_idxs = batch
        scores = self.bienc(*model_input)
        mask = torch.full_like(scores, False, dtype=torch.bool)
        mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
        mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
        mask.fill_diagonal_(False)
        loss = self.loss_fn(scores, mask)

        scores = scores.cpu().numpy()

        bs = scores.shape[0]
        preds = np.argmax(scores, axis=1)
        acc = np.sum(preds == np.arange(bs)).astype(float) / bs

        return {"acc": acc, "loss": loss}