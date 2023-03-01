from collections import defaultdict
from typing import Dict, Union, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from neptune.new import Run
from torch.optim import Optimizer, AdamW

from bienc.gen_cross import gen_cross_df
from bienc.inference import embed_and_nn, entities_inference, filter_languages, predict_entities
from bienc.metrics import get_bienc_thresh_metrics, get_avg_precision_threshs, log_dct, get_log_mir_metrics, \
    get_bienc_cands_metrics, get_average_precision_cands, BIENC_STANDALONE_THRESHS
from bienc.model import BiencoderModule, Biencoder
from config import CFG
from metrics import get_fscore
from utils import get_learning_rate_momentum, are_entity_ids_aligned, get_topic_id_gold


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
        self.run["loss"].log(loss.item(), step=self.global_step)
        lr, momentum = get_learning_rate_momentum(self.optimizers().optimizer)
        self.run["lr"].log(lr, step=self.global_step)
        if momentum:
            self.run["momentum"].log(momentum, step=self.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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

    def validation_epoch_end(self, outputs):
        if CFG.tune_lr or CFG.tune_bs:
            return
        dct = agg_outputs(outputs)
        print(f"Evaluation in-batch accuracy: {dct['acc']:.5}")
        print(f"Evaluation loss: {dct['loss']:.5}")
        self.run["val/acc"].log(dct['acc'], step=self.global_step)
        self.run["val/loss"].log(dct['loss'], step=self.global_step)

    def on_train_epoch_end(self):
        if self.folds == "no" or CFG.tune_lr or CFG.tune_bs:
            return
        print(f"Running inference-mode evaluation for epoch {self.current_epoch}...")
        optim, cross_df = wrap_evaluate_inference(self.bienc, self.device, CFG.batch_size,
                                                  self.val_corr_df, self.topic2text, self.content2text, self.c2i,
                                                  self.optimizers().optimizer,
                                                  CFG.FILTER_LANG, self.t2lang, self.c2lang,
                                                  self.current_epoch == CFG.num_epochs - 1,
                                                  self.global_step, self.run)
        if self.current_epoch == CFG.num_epochs - 1:
            (self.cross_output_dir / f"{self.experiment_id}").mkdir(parents=True, exist_ok=True)
            cross_df_fname = self.cross_output_dir / f"{self.experiment_id}" / f"fold-{self.fold_idx}.csv"
            cross_df.to_csv(cross_df_fname, index=False)
            print(f"Wrote cross df to {cross_df_fname}")


def evaluate_inference(encoder: BiencoderModule, device: torch.device, batch_size: int, corr_df: pd.DataFrame,
                       topic2text: Dict[str, str], content2text: Dict[str, str], e2i: Dict[str, int],
                       filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
                       gen_cross: bool,
                       global_step: int, run: Run) -> Union[None, pd.DataFrame]:
    """Evaluates inference mode."""
    # Make sure entity idxs align
    entity_ids = sorted(list(content2text.keys()))
    assert are_entity_ids_aligned(entity_ids, e2i)

    # Prepare nearest neighbors data structure for entities
    nn_model = embed_and_nn(encoder, entity_ids, content2text, CFG.NUM_NEIGHBORS, batch_size, device)

    # Get nearest neighbor distances and indices
    data_ids = sorted(list(set(corr_df["topic_id"])))
    distances, indices = entities_inference(data_ids, encoder, nn_model, topic2text, device, batch_size)

    # Filter languages
    if filter_lang:
        e2i = e2i.copy()
        e2i["dummy"] = -1
        distances, indices = filter_languages(distances, indices, data_ids, e2i, t2lang, c2lang)

    # Metrics
    t2gold = get_topic_id_gold(corr_df)

    # Thresh metrics
    precision_dct, recall_dct, micro_prec_dct, pcr_dct = get_bienc_thresh_metrics(distances, indices, data_ids, e2i, t2gold)
    avg_precision = get_avg_precision_threshs(distances, indices, data_ids, e2i, t2gold)
    print(f"Mean average precision (thresh) @ {CFG.NUM_NEIGHBORS}: {avg_precision:.5}")
    run["val/avg_precision"].log(avg_precision, step=global_step)
    log_dct(precision_dct, "val/precision", global_step, run)
    log_dct(recall_dct, "val/recall", global_step, run)
    log_dct(micro_prec_dct, "val/micro_precision", global_step, run)
    log_dct(pcr_dct, "val/pcr", global_step, run)

    # Cands metrics
    get_log_mir_metrics(indices, data_ids, e2i, t2gold, global_step, run)
    precision_dct, recall_dct, micro_prec_dct, pcr_dct = get_bienc_cands_metrics(indices, data_ids, e2i, t2gold, 100)
    avg_precision = get_average_precision_cands(indices, data_ids, e2i, t2gold)
    print(f"Mean average precision (cands) @ {CFG.NUM_NEIGHBORS}: {avg_precision:.5}")
    run["cands/avg_precision"].log(avg_precision, step=global_step)
    log_dct(precision_dct, "cands/precision", global_step, run)
    log_dct(recall_dct, "cands/recall", global_step, run)
    log_dct(micro_prec_dct, "cands/micro_precision", global_step, run)
    log_dct(pcr_dct, "cands/pcr", global_step, run)

    # Thresholds
    best_thresh = None
    best_fscore = -1.0
    thresh2score = {}
    for thresh in BIENC_STANDALONE_THRESHS:
        t2preds = predict_entities(data_ids, distances, indices, thresh, e2i)
        fscore = get_fscore(t2gold, t2preds)
        thresh2score[thresh] = fscore
        if fscore > best_fscore:
            best_fscore = fscore
            best_thresh = thresh
        print(f"validation f2 using threshold {thresh}: {fscore:.5}")
        run[f"val/F2@{thresh}"].log(fscore, step=global_step)
    print(f"Best threshold: {best_thresh}")
    print(f"Best F2 score: {best_fscore}")
    run[f"val/best_thresh"].log(best_thresh, step=global_step)
    run[f"val/best_F2"].log(best_fscore, step=global_step)

    # Generate cross df
    if gen_cross:
        cross_df = gen_cross_df(distances, indices, corr_df, e2i)
        return cross_df


def wrap_evaluate_inference(model: Biencoder, device: torch.device, batch_size: int, corr_df: pd.DataFrame,
                            topic2text: Dict[str, str], content2text: Dict[str, str], e2i: Dict[str, int],
                            optim: Optimizer,
                            filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
                            gen_cross: bool,
                            global_step: int, run: Run) -> Tuple[Optimizer, Union[None, pd.DataFrame]]:
    optimizer_state_dict = optim.state_dict()
    cross_df = evaluate_inference(model.topic_encoder, device, batch_size,
                                  corr_df, topic2text, content2text, e2i,
                                  filter_lang, t2lang, c2lang,
                                  gen_cross,
                                  global_step, run)
    optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
    optim.load_state_dict(optimizer_state_dict)
    return optim, cross_df
