import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from config import CFG
from cross.metrics import get_cross_f2, log_fscores
from utils import get_learning_rate_momentum


def agg_outputs(outputs, dset_len):
    loss = 0.0
    all_probs = torch.empty(dset_len)
    i = 0
    for out in outputs:
        loss += out["loss"]
        probs = out["probs"]
        bs = probs.shape[0]
        all_probs[i:(i+bs)] = probs
        i += bs
    loss = loss / len(outputs)
    return loss, all_probs.numpy()


class LitCross(pl.LightningModule):
    def __init__(self, model, loss_fn, get_train_loader,
                 val_corr_df,
                 learning_rate, weigth_decay, batch_size,
                 run):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.get_train_loader = get_train_loader
        self.val_corr_df = val_corr_df
        self.learning_rate = learning_rate
        self.weight_decay = weigth_decay
        self.batch_size = batch_size
        self.run = run

        if not (CFG.tune_bs or CFG. tune_lr):
            self.automatic_optimization = False
            self.training_step = self.training_step_manual
        else:
            self.training_step = self.training_step_automatic

    def train_dataloader(self):
        return self.get_train_loader(self.batch_size)

    def training_step_automatic(self, batch, batch_idx):
        *model_input, labels = batch
        logits = self.model(*model_input)
        loss = self.loss_fn(logits, labels)

        # Log
        if not CFG.tune_bs:
            self.run["loss"].log(loss.item(), step=self.global_step)
            lr, momentum = get_learning_rate_momentum(self.optimizers().optimizer)
            self.run["lr"].log(lr, step=self.global_step)
            if momentum:
                self.run["momentum"].log(momentum, step=self.global_step)

        return loss

    def training_step_manual(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        *model_input, labels = batch
        logits = self.model(*model_input)
        loss = self.loss_fn(logits, labels)

        self.manual_backward(loss)
        opt.step()

        # Log
        self.run["loss"].log(loss.item(), step=self.global_step)
        lr, momentum = get_learning_rate_momentum(self.optimizers().optimizer)
        self.run["lr"].log(lr, step=self.global_step)
        if momentum:
            self.run["momentum"].log(momentum, step=self.global_step)

        # Scheduler
        if CFG.scheduler == "cosine":
            sched = self.lr_schedulers()
            sched.step()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if CFG.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2)
        elif CFG.scheduler == "cosine":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.num_epochs * len(self.get_train_loader(self.batch_size)))
        else:
            assert CFG.scheduler == "none"
            return optimizer
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        *model_input, labels = batch
        logits = self.model(*model_input)
        loss = self.loss_fn(logits, labels)
        probs = logits.softmax(dim=1)[:, 1]

        return {"loss": loss.item(), "probs":  probs.cpu().reshape(-1)}

    def validation_epoch_end(self, outputs):
        if CFG.tune_lr or CFG.tune_bs:
            return
        loss, probs = agg_outputs(outputs, len(self.trainer.val_dataloaders[0].dataset))

        print(f"Evaluation loss: {loss:.5}")
        self.run["cross/loss"].log(loss, step=self.global_step)

        fscores = get_cross_f2(probs, self.val_corr_df)
        log_fscores(fscores, self.global_step, self.run)

    # TODO
    # def training_epoch_end(self, outputs):
    #     # Save checkpoint
    #     if CFG.checkpoint:
    #         save_checkpoint(checkpoint_dir / f"{run_id}" / f"epoch-{epoch}.pt", global_step,
    #                         model.state_dict(), optim.state_dict(), None, scaler.state_dict())
