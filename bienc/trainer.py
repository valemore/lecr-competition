from collections import defaultdict
from typing import Dict, Union, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from neptune.new import Run
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from bienc.gen_cands import get_cand_df
from bienc.inference import do_nn
from bienc.metrics import log_dct, get_log_mir_metrics, get_bienc_cands_metrics, get_average_precision_cands
from bienc.model import BiencoderModule, Biencoder
from config import CFG
from utils import get_learning_rate_momentum, are_content_ids_aligned, get_topic_id_gold


def agg_outputs(outputs):
    dct = defaultdict(lambda: 0.0)
    for out in outputs:
        for k, v in out.items():
            dct[k] = dct[k] + v
    for k, v in dct.items():
        dct[k] = v / len(outputs)
    return dct


class LitBienc(pl.LightningModule):
    def __init__(self, bienc, loss_fn, get_train_loader,
                 topic2text, content2text, c2i, t2lang, c2lang,
                 learning_rate, weigth_decay, batch_size,
                 cross_output_dir, experiment_id,
                 folds, fold_idx, val_corr_df, run):
        super().__init__()
        self.bienc = bienc
        self.loss_fn = loss_fn
        self.get_train_loader = get_train_loader
        self.val_corr_df = val_corr_df
        self.topic2text = topic2text
        self.content2text = content2text
        self.c2i = c2i
        self.t2lang = t2lang
        self.c2lang = c2lang
        self.learning_rate = learning_rate
        self.weight_decay = weigth_decay
        self.batch_size = batch_size
        self.cross_output_dir = cross_output_dir
        self.experiment_id = experiment_id
        self.folds = folds
        self.fold_idx = fold_idx
        self.run = run

        if not (CFG.tune_bs or CFG. tune_lr):
            self.automatic_optimization = False
            self.training_step = self.training_step_manual
        else:
            self.training_step = self.training_step_automatic

    def train_dataloader(self):
        return self.get_train_loader(self.batch_size)

    def training_step_automatic(self, batch, batch_idx):
        *model_input, topic_idxs, content_idxs = batch

        scores = self.bienc(*model_input)
        mask = torch.full_like(scores, False, dtype=torch.bool)
        mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
        mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
        mask.fill_diagonal_(False)
        loss = self.loss_fn(scores, mask)

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

        *model_input, topic_idxs, content_idxs = batch
        scores = self.bienc(*model_input)
        mask = torch.full_like(scores, False, dtype=torch.bool)
        mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
        mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
        mask.fill_diagonal_(False)
        loss = self.loss_fn(scores, mask)

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=CFG.clip, gradient_clip_algorithm="norm")
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

    def validation_epoch_end(self, outputs):
        dct = agg_outputs(outputs)
        print(f"Evaluation in-batch accuracy: {dct['acc']:.5}")
        print(f"Evaluation loss: {dct['loss']:.5}")
        self.run["val/acc"].log(dct['acc'], step=self.global_step)
        self.run["val/loss"].log(dct['loss'], step=self.global_step)

    def on_validation_epoch_end(self):
        print(f"Running inference-mode evaluation for epoch {self.current_epoch}...")
        optim, cross_df, avg_precision = wrap_evaluate_inference(self.bienc, self.device, CFG.batch_size,
                                                                 self.val_corr_df, self.topic2text, self.content2text, self.c2i,
                                                                 self.optimizers().optimizer,
                                                                 CFG.FILTER_LANG, self.t2lang, self.c2lang,
                                                                 self.current_epoch == CFG.num_epochs - 1,
                                                                 self.global_step, self.run)

        if CFG.scheduler == "plateau":
            sched = self.lr_schedulers()
            sched.step(avg_precision)

        if self.current_epoch == CFG.num_epochs - 1:
            (self.cross_output_dir / f"{self.experiment_id}").mkdir(parents=True, exist_ok=True)
            cross_df_fname = self.cross_output_dir / f"{self.experiment_id}" / f"fold-{self.fold_idx}.csv"
            cross_df.to_csv(cross_df_fname, index=False)
            print(f"Wrote cross df to {cross_df_fname}")


def evaluate_inference(encoder: BiencoderModule, device: torch.device, batch_size: int, corr_df: pd.DataFrame,
                       topic2text: Dict[str, str], content2text: Dict[str, str], c2i: Dict[str, int],
                       filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
                       gen_cross: bool,
                       global_step: int, run: Run) -> Tuple[Union[None, pd.DataFrame], float]:
    """Evaluates inference mode."""
    # Make sure entity idxs align
    content_ids = sorted(list(content2text.keys()))
    assert are_content_ids_aligned(content_ids, c2i)

    topic_ids = sorted(list(set(corr_df["topic_id"])))
    indices = do_nn(encoder, topic_ids, content_ids, topic2text, content2text,
                    filter_lang, t2lang, c2lang, c2i,
                    batch_size, device)

    # Metrics
    t2gold = get_topic_id_gold(corr_df)

    # Cands metrics
    get_log_mir_metrics(indices, topic_ids, c2i, t2gold, global_step, run)
    precision_dct, recall_dct, micro_prec_dct, pcr_dct = get_bienc_cands_metrics(indices, topic_ids, c2i, t2gold, CFG.NUM_NEIGHBORS)
    avg_precision = get_average_precision_cands(indices, topic_ids, c2i, t2gold)
    print(f"Mean average precision (cands) @ {CFG.NUM_NEIGHBORS}: {avg_precision:.5}")
    run["cands/avg_precision"].log(avg_precision, step=global_step)
    log_dct(precision_dct, "cands/precision", global_step, run)
    log_dct(recall_dct, "cands/recall", global_step, run)
    log_dct(micro_prec_dct, "cands/micro_precision", global_step, run)
    log_dct(pcr_dct, "cands/pcr", global_step, run)

    # Generate cross df
    if gen_cross:
        cross_df = (get_cand_df(topic_ids, indices, c2i, CFG.MAX_NUM_CANDS)
                        .merge(corr_df.loc[:, ["topic_id", "content_ids"]], on="topic_id", how="left")
                        .loc[:, ["topic_id", "content_ids", "cands"]])
    else:
        cross_df = None
    return cross_df, avg_precision


def wrap_evaluate_inference(model: Biencoder, device: torch.device, batch_size: int, corr_df: pd.DataFrame,
                            topic2text: Dict[str, str], content2text: Dict[str, str], e2i: Dict[str, int],
                            optim: Optimizer,
                            filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
                            gen_cross: bool,
                            global_step: int, run: Run) -> Tuple[Optimizer, Union[None, pd.DataFrame], float]:
    optimizer_state_dict = optim.state_dict()
    cross_df, avg_precision = evaluate_inference(model.topic_encoder, device, batch_size,
                                                 corr_df, topic2text, content2text, e2i,
                                                 filter_lang, t2lang, c2lang,
                                                 gen_cross,
                                                 global_step, run)
    optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
    optim.load_state_dict(optimizer_state_dict)
    return optim, cross_df, avg_precision
