from datetime import datetime
from collections import defaultdict
from pathlib import Path
import random

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

from bienc.inference import embed_topics_nn, bienc_inference, predict_topics
from bienc.typehints import LossFunction
from config import DATA_DIR, VAL_SPLIT_SEED, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, SCORE_FN, NUM_WORKERS, NUM_NEIGHBORS
from data.content import get_content2text
from bienc.dset import BiencDataset
from data.topics import get_topic2text
from bienc.model import Biencoder, BiencoderModule
from bienc.losses import BidirectionalMarginLoss
from metrics import get_fscore
from utils import get_learning_rate_momentum, log_recall_dct, \
    flatten_content_ids, get_content_id_gold, are_topics_aligned, get_topic_id_gold
from bienc.metrics import get_recall_dct, get_min_max_ranks, get_mean_inverse_rank


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
             run: Run) -> dict[str, float]:
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
                       topic2text: dict[str, str], content2text: dict[str, str], t2i: dict[str, int],
                       global_step: int, run: Run) -> None:
    """Evaluates inference mode."""
    # Make sure topic idxs align
    topic_ids = sorted(list(set(corr_df["topic_id"])))
    assert are_topics_aligned(topic_ids, t2i)

    # Prepare neirest neighbors data structure for topics
    nn_model = embed_topics_nn(encoder, topic_ids, topic2text, NUM_NEIGHBORS, batch_size, device)

    # Get nearest neighbor distances and indices
    content_ids = flatten_content_ids(corr_df)
    distances, indices = bienc_inference(content_ids, encoder, nn_model, content2text, device, batch_size)

    get_log_rank_metrics(indices, content_ids, t2i, corr_df, global_step, run)

    t2gold = get_topic_id_gold(corr_df)
    best_thresh = None
    best_fscore = -1.0
    for thresh in np.arange(0.01, 1.01, 0.01):
        c2preds = predict_topics(content_ids, distances, indices, thresh, t2i)
        t2preds = post_process(c2preds, indices, t2i)
        fscore = get_fscore(t2gold, t2preds)
        if fscore > best_fscore:
            best_fscore = fscore
            best_thresh = thresh
        print(f"validation f2 using threshold {thresh}: {fscore:.5}")
        run[f"val/f2@{thresh}"].log(fscore, step=global_step)
    print(f"Best threshold: {best_thresh}")
    run[f"val/best_thresh"].log(best_thresh, step=global_step)


def post_process(c2preds, indices, t2i):
    t2preds = defaultdict(set)
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(sorted(c2preds))}
    i2c = {content_idx: content_id for content_id, content_idx in c2i.items()}
    for content_id, topic_ids in c2preds.items():
        for topic_id in topic_ids:
            t2preds[topic_id].add(content_id)
    for topic_id in t2i:
        content_ids = t2preds[topic_id]
        if len(content_ids) == 0:
            topic_idx = t2i[topic_id]
            matches = np.argwhere(indices == topic_idx) # shape (num_matches, 2) -  last dimension distinguishes between content_idx and topic_idx
            content_idx = np.argmin(matches, axis=0)[1]
            t2preds[topic_id] = {i2c[content_idx]}
    return t2preds


def get_log_rank_metrics(indices,
                         content_ids: list[str], t2i: dict[str, int], corr_df: pd.DataFrame,
                         global_step: int, run: Run) -> None:
    """Compare with gold, compute and log rank metrics."""
    c2gold = get_content_id_gold(corr_df)
    min_ranks, max_ranks = get_min_max_ranks(indices, content_ids, c2gold, t2i)
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
    log_recall_dct(min_recall_dct, global_step, run, "val_min")
    log_recall_dct(max_recall_dct, global_step, run, "val_max")


def main(tiny=False,
         debug=False,
         batch_size=128,
         max_lr=3e-5,
         weight_decay=0.0,
         margin=6.0,
         num_epochs=2,
         use_amp=True,
         experiment_name="first",
         all_folds=False,
         output_dir="../out"):
    device = torch.device("cuda") if (not debug) and torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(output_dir)

    content_df = pd.read_csv(DATA_DIR / "content.csv")
    corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
    topics_df = pd.read_csv(DATA_DIR / "topics.csv")

    if tiny:
        corr_df = corr_df.iloc[:1000, :].reset_index(drop=True)

    topics_in_scope = sorted(list(set(corr_df["topic_id"])))
    random.seed(VAL_SPLIT_SEED)
    random.shuffle(topics_in_scope)

    fold_idx = 0
    for topics_in_scope_train_idxs, topics_in_scope_val_idxs in KFold(n_splits=5).split(topics_in_scope):
        if not all_folds and fold_idx > 0:
            break
        train_topics = set(topics_in_scope[idx] for idx in topics_in_scope_train_idxs)
        val_topics = set(topics_in_scope[idx] for idx in topics_in_scope_val_idxs)
        train_corr_df = corr_df.loc[corr_df["topic_id"].isin(train_topics), :].reset_index(drop=True)
        val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)

        train_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(train_corr_df["topic_id"]))))}
        val_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(val_corr_df["topic_id"]))))}

        topic2text = get_topic2text(topics_df)
        content2text = get_content2text(content_df)

        train_dset = BiencDataset(train_corr_df["topic_id"], train_corr_df["content_ids"],
                                  topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, train_t2i)
        val_dset = BiencDataset(val_corr_df["topic_id"], val_corr_df["content_ids"],
                                topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, val_t2i)

        model = Biencoder(SCORE_FN).to(device)
        loss_fn = BidirectionalMarginLoss(device, margin)

        train_loader = DataLoader(train_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
        optim = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

        scaler = GradScaler(enabled=use_amp)

        # Prepare logging and saving
        run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["src/**/*.py", "src/*.py"],
            tags=[experiment_name] + [f"fold{fold_idx}"] + (["TINY"] if tiny else []) + (["DEBUG"] if debug else []))

        # Train
        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            print(f"Training epoch {epoch}...")
            global_step = train_one_epoch(model, loss_fn, train_loader, device, optim, None, use_amp, scaler,
                                          global_step, run)

            # Loss and in-batch accuracy for training validation set
            print(f"Running in-batch evaluation for epoch {epoch}...")
            evaluate(model, loss_fn, val_loader, device, global_step, run)

            # Evaluate inference
            print(f"Running inference-mode evaluation for epoch {epoch}...")
            evaluate_inference(model.topic_encoder, device, batch_size, val_corr_df, topic2text, content2text, val_t2i,
                               global_step, run)

        # Save artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.topic_encoder.state_dict(), output_dir / f"{run_start}.pt")

        fold_idx += 1


if __name__ == "__main__":
    main()
