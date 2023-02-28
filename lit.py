import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neptune.new import Run

from config import CFG
from train_bienc import wrap_evaluate_inference
from utils import get_learning_rate_momentum


class LitBienc(pl.LightningModule):
    def __init__(self,
                 bienc, loss_fn,
                 val_corr_df, topic2text, content2text, c2i, t2lang, c2lang,
                 learning_rate, weigth_decay,
                 cross_output_dir,
                 experiment_id, fold_idx, run):
        super().__init__()
        self.bienc = bienc
        self.loss_fn = loss_fn
        self.val_corr = val_corr_df
        self.topic2text = topic2text
        self.content2text = content2text
        self.c2i = c2i
        self.t2lang = t2lang
        self.c2lang = c2lang
        self.max_lr = learning_rate
        self.weight_decay = weigth_decay
        self.cross_output_dir = cross_output_dir
        self.experiment_id = experiment_id
        self.fold_idx = fold_idx
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

        return {"acc": acc.item(), "loss": loss.item()}

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()  # free memory
        print(f"Running inference-mode evaluation for epoch {self.current_epoch}...")
        optim, cross_df = wrap_evaluate_inference(self.model, self.device, CFG.batch_size,
                                                  self.val_corr_df, self.topic2text, self.content2text, self.c2i,
                                                  self.optimizers()[0]._optimizer,
                                                  CFG.FILTER_LANG, self.t2lang, self.c2lang,
                                                  self.current_epoch == CFG.num_epochs - 1,
                                                  self.global_step, self.run)
        if self.current_epoch == CFG.num_epochs - 1:
            (self.cross_output_dir / f"{self.experiment_id}").mkdir(parents=True, exist_ok=True)
            cross_df_fname = self.cross_output_dir / f"{self.experiment_id}" / f"fold-{self.fold_idx}.csv"
            cross_df.to_csv(cross_df_fname, index=False)
            print(f"Wrote cross df to {cross_df_fname}")
