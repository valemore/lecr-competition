from datetime import datetime
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from neptune.new import Run
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import neptune.new as neptune

from bienc.inference import embed_and_nn, entities_inference, predict_entities
import bienc.tokenizer as tokenizer
from bienc.typehints import LossFunction
from config import DATA_DIR, VAL_SPLIT_SEED, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, SCORE_FN, NUM_WORKERS, NUM_NEIGHBORS
from data.content import get_content2text
from bienc.dset import BiencDataset
from data.topics import get_topic2text
from bienc.model import Biencoder, BiencoderModule
from bienc.losses import BidirectionalMarginLoss
from metrics import get_fscore
from typehints import MetricDict
from utils import get_learning_rate_momentum, flatten_content_ids, are_entity_ids_aligned, get_topic_id_gold
from bienc.metrics import get_recall_dct, get_min_max_ranks, get_mean_inverse_rank


tokenizer.init_tokenizer()


def train_one_epoch(model: Biencoder, loss_fn: LossFunction, train_loader: DataLoader, device: torch.device,
                    optim: Optimizer, scheduler, use_amp: bool, scaler: GradScaler, global_step: int, run: Run) -> int:
    """Train one epoch of Bi-encoder."""
    step = global_step
    model.train()
    for batch in tqdm(train_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, topic_idxs = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            scores = model(*model_input)
            mask = torch.full_like(scores, False, dtype=torch.bool)
            mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
            mask.fill_diagonal_(False)
            loss = loss_fn(scores, mask)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # Log
        run["train/loss"].log(loss.item(), step=step)
        lr, momentum = get_learning_rate_momentum(optim)
        run["lr"].log(lr, step=step)
        if momentum:
            run["momentum"].log(momentum, step=step)

        step += 1
    return step


def evaluate(model: Biencoder, loss_fn: LossFunction, val_loader: DataLoader, device: torch.device, global_step: int,
             run: Run) -> Dict[str, float]:
    """Performs in-batch validation."""
    acc_cumsum = 0.0
    loss_cumsum = 0.0
    num_examples = 0
    num_batches = 0

    model.eval()
    for batch in tqdm(val_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, entity_idxs = batch
        with torch.no_grad():
            scores = model(*model_input)
            mask = torch.full_like(scores, False, dtype=torch.bool)
            mask[entity_idxs.unsqueeze(-1) == entity_idxs.unsqueeze(0)] = True
            mask.fill_diagonal_(False)
            loss = loss_fn(scores, mask)

        scores = scores.cpu().numpy()

        bs = scores.shape[0]
        preds = np.argmax(scores, axis=1)

        acc_cumsum += np.sum(preds == np.arange(bs))
        loss_cumsum += loss.item()
        num_examples += bs
        num_batches += 1

    acc = acc_cumsum / num_examples
    loss = loss_cumsum / num_batches

    print(f"Evaluation in-batch accuracy: {acc:.5}")
    print(f"Evaluation loss: {loss:.5}")

    run["val/acc"].log(acc, step=global_step)
    run["val/loss"].log(loss, step=global_step)

    return {"acc": acc, "loss": loss}


def evaluate_inference(encoder: BiencoderModule, device: torch.device, batch_size: int, corr_df: pd.DataFrame,
                       topic2text: Dict[str, str], content2text: Dict[str, str], e2i: Dict[str, int],
                       global_step: int, run: Run) -> None:
    """Evaluates inference mode."""
    # Make sure entity idxs align
    entity_ids = sorted(list(content2text.keys()))
    assert are_entity_ids_aligned(entity_ids, e2i)

    # Prepare nearest neighbors data structure for entities
    nn_model = embed_and_nn(encoder, entity_ids, content2text, NUM_NEIGHBORS, batch_size, device)

    # Get nearest neighbor distances and indices
    data_ids = sorted(list(set(corr_df["topic_id"])))
    distances, indices = entities_inference(data_ids, encoder, nn_model, topic2text, device, batch_size)

    # Rank metrics
    t2gold = get_topic_id_gold(corr_df)
    get_log_rank_metrics(indices, data_ids, e2i, t2gold, global_step, run)
    precision_dct, recall_dct, avg_precision_dct = get_precision_recall_metrics(indices, data_ids, e2i, t2gold)
    print(f"Mean average precision @ 100: {avg_precision_dct[100]:.5}")
    log_precision_dct(precision_dct, "val/precision", global_step, run)
    log_precision_dct(recall_dct, "val/recall", global_step, run)
    log_precision_dct(avg_precision_dct, "val/avg_precision", global_step, run)

    # Thresholds
    best_thresh = None
    best_fscore = -1.0
    thresh2score = {}
    for thresh in np.arange(0.1, 0.32, 0.04):
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


def get_precision_recall_metrics(indices, topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> Tuple[MetricDict, MetricDict, MetricDict]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    tp = np.empty_like(indices, dtype=int) # mask indicating whether prediction is a true positive
    num_gold = np.empty(len(topic_ids), dtype=int) # how many content ids are in gold?
    for i, (idxs, topic_id) in enumerate(zip(indices, topic_ids)):
        gold = t2gold[topic_id]
        tp[i, :] = np.array([int(i2e[idx] in gold) for idx in idxs], dtype=int)
        num_gold[i] = len(gold)

    precision_dct = {num_cands: 0.0 for num_cands in range(1, NUM_NEIGHBORS + 1)}
    recall_dct = {num_cands: 0.0 for num_cands in range(1, NUM_NEIGHBORS + 1)}
    avg_precision_dct = {num_cands: 0.0 for num_cands in range(1, NUM_NEIGHBORS + 1)}

    acc_tp = np.zeros(len(topic_ids), dtype=float) # accumulating true positives for all topic ids
    acc_avg_prec = np.zeros(len(topic_ids), dtype=float) # accumulating average precisino for all topic ids
    prev_rec = np.zeros(len(topic_ids), dtype=float) # previous recall
    for j, num_cands in enumerate(range(1, NUM_NEIGHBORS + 1)):
        acc_tp += tp[:, j]
        prec = acc_tp / num_cands
        rec = acc_tp / num_gold
        acc_avg_prec += prec * (rec - prev_rec)
        prev_rec = rec
        precision_dct[num_cands] = np.mean(prec)
        recall_dct[num_cands] = np.mean(rec)
        avg_precision_dct[num_cands] = np.mean(acc_avg_prec)
    return precision_dct, recall_dct, avg_precision_dct


def log_precision_dct(dct: Dict[int, float], label: str, global_step: int, run: Run):
    for k, v in dct.items():
        run[f"{label}@{k}"].log(v, step=global_step)


def log_recall_dct(recall_dct: Dict[int, float], global_step: int, run: Run, label: str) -> None:
    """Log a recall dictionary to neptune.ai"""
    for k, v in recall_dct.items():
        run[f"{label}@{k}"].log(v, step=global_step)


def get_log_rank_metrics(indices,
                         data_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]],
                         global_step: int, run: Run) -> None:
    """Compare with gold, compute and log rank metrics."""
    min_ranks, max_ranks = get_min_max_ranks(indices, data_ids, t2gold, e2i)
    min_mir = get_mean_inverse_rank(min_ranks)
    max_mir = get_mean_inverse_rank(max_ranks)
    min_recall_dct = get_recall_dct(min_ranks)
    max_recall_dct = get_recall_dct(max_ranks)

    print(f"Evaluation inference mode mean inverse min rank: {min_mir:.5}")
    print(f"Evaluation inference mode min recall@1: {min_recall_dct[1]:.5}")
    print(f"Evaluation inference mode mean inverse max rank: {max_mir:.5}")
    print(f"Evaluation inference mode max recall@1: {max_recall_dct[1]:.5}")

    run["val/min_mir"].log(min_mir, step=global_step)
    run["val/max_mir"].log(max_mir, step=global_step)
    log_recall_dct(min_recall_dct, global_step, run, "val/min_recall")
    log_recall_dct(max_recall_dct, global_step, run, "val/max_recall")


def main(tiny=False,
         batch_size=256,
         max_lr=3e-5,
         weight_decay=0.0,
         margin=6.0,
         num_epochs=5,
         use_amp=True,
         experiment_name="full",
         folds="first", # "first", "all", "no"
         output_dir="../out"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(output_dir)

    content_df = pd.read_csv(DATA_DIR / "content.csv")
    corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
    topics_df = pd.read_csv(DATA_DIR / "topics.csv")

    if tiny:
        corr_df = corr_df.sample(1000).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(
            set(flatten_content_ids(corr_df)) | set(content_df["id"].sample(1000))), :].reset_index(drop=True)

    topics_in_scope = sorted(list(set(corr_df["topic_id"])))
    random.seed(VAL_SPLIT_SEED)
    random.shuffle(topics_in_scope)

    fold_idx = 0 if folds != "no" else -1
    for topics_in_scope_train_idxs, topics_in_scope_val_idxs in KFold(n_splits=5).split(topics_in_scope):
        if (folds == "first" and fold_idx > 0) or (folds == "no" and fold_idx == 0):
            break
        if folds != "no":
            train_topics = set(topics_in_scope[idx] for idx in topics_in_scope_train_idxs)
        else:
            train_topics = topics_in_scope
        train_corr_df = corr_df.loc[corr_df["topic_id"].isin(train_topics), :].reset_index(drop=True)
        train_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(train_corr_df["topic_id"]))))}

        c2i = {content_id: content_idx for content_idx, content_id in enumerate(sorted(set(content_df["id"])))}
        topic2text = get_topic2text(topics_df)
        content2text = get_content2text(content_df)

        train_dset = BiencDataset(train_corr_df["topic_id"], train_corr_df["content_ids"],
                                  topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, train_t2i)
        train_loader = DataLoader(train_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True)

        if folds != "no":
            val_topics = set(topics_in_scope[idx] for idx in topics_in_scope_val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(val_corr_df["topic_id"]))))}
            val_dset = BiencDataset(val_corr_df["topic_id"], val_corr_df["content_ids"],
                                    topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, val_t2i)
            val_loader = DataLoader(val_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)

        model = Biencoder(SCORE_FN).to(device)
        model = torch.nn.DataParallel(model)
        loss_fn = BidirectionalMarginLoss(device, margin)

        optim = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        scaler = GradScaler(enabled=use_amp)

        # Prepare logging and saving
        run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["src/**/*.py", "src/*.py"],
            tags=[experiment_name] + [f"fold{fold_idx}"] + (["TINY"] if tiny else []))

        # Train
        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            print(f"Training epoch {epoch}...")
            global_step = train_one_epoch(model, loss_fn, train_loader, device, optim, None, use_amp, scaler,
                                          global_step, run)

            if folds != "no":
                # Loss and in-batch accuracy for training validation set
                print(f"Running in-batch evaluation for epoch {epoch}...")
                evaluate(model, loss_fn, val_loader, device, global_step, run)

                # Evaluate inference
                print(f"Running inference-mode evaluation for epoch {epoch}...")
                evaluate_inference(model.topic_encoder, device, batch_size, val_corr_df, topic2text, content2text, c2i,
                                   global_step, run)

        # Save artifacts
        (output_dir / f"{experiment_name}_{run_start}" / "bienc").mkdir(parents=True, exist_ok=True)
        (output_dir / f"{experiment_name}_{run_start}" / "tokenizer").mkdir(parents=True, exist_ok=True)
        model.content_encoder.encoder.save_pretrained(output_dir / f"{experiment_name}_{run_start}" / "bienc")
        tokenizer.tokenizer.save_pretrained(output_dir / f"{experiment_name}_{run_start}" / "tokenizer")

        fold_idx += 1


if __name__ == "__main__":
    main()
