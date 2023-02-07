# Train script for Bi-encoder
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import neptune.new as neptune
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.bienc import Biencoder
from model.losses import BidirectionalMarginLoss
from data.dset import BiencDataset

from utils import get_learning_rate_momentum, save_checkpoint, get_recall_dct, cache, log_recall_dct
from config import CONTENT_NUM_TOKENS, TOPIC_NUM_TOKENS, VAL_SPLIT_SEED, SCORE_FN, DATA_DIR

TINY = False
DEBUG = False
ARTIFACTS_DIR = Path.home() / "artifacts"
CACHE_DIR = Path.home() / "cache"
REFRESH_CACHE = True


def full_eval(query_encoder, query_loader, entity_loader, score_fn, device):
    """Evaluates QUERY_ENCODER on QUERY_LOADER by comparing embeddings to all entities in ENTITY_LOADER using SCORE_FN."""
    all_scores = []
    all_labels = []

    query_encoder.eval()
    for batch in tqdm(query_loader):
        batch = tuple(x.to(device) for x in batch)
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            scores_lst = []
            query_emb = query_encoder(input_ids, attention_mask)
            for ent_emb in entity_loader:
                ent_emb = ent_emb.to(device)
                scores = score_fn(query_emb, ent_emb)
                scores_lst.append(scores.detach().cpu().numpy())

            all_scores.append(np.concatenate(scores_lst, axis=1))
        all_labels.append(labels.detach().cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_scores, all_labels


def evaluate(model, val_loader, device, loss_fn, global_step, run):
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

        scores = scores.detach().cpu().numpy()

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


def train_one_epoch(model, train_loader, device, loss_fn, optim, scheduler, use_amp, scaler, global_step, run):
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
        run["momentum"].log(momentum, step=step)

        step += 1
    return step


# Main
def main(num_epochs = 5,
         batch_size = 512,
         max_lr = 1e-4,
         weight_decay = 0.0,
         margin = 6.0,
         use_amp=True,
         min_count = 2,
         min_rel = 0.3,
         output_dir=str(Path.home() / "out"),
         experiment_name="first",
         save_outputs=False):

    device = torch.device("cuda") if (not DEBUG) and torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
    checkpoint_dir = output_dir / f"{experiment_name}_{run_start}"
    checkpoint_dir.mkdir(exist_ok=True)

    # Data preparation
    content_df = pd.read_csv(DATA_DIR / "content.csv")
    corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
    topics_df = pd.read_csv(DATA_DIR / "topics.csv")


    if TINY:
        df = df.sample(10000).reset_index(drop=True)

    mdl_cache = MDLCache(tokenizer, mdl_parser, full_entity_universe, CONTENT_NUM_TOKENS)
    quenc = QueryEncoder(tokenizer, TOPIC_NUM_TOKENS)

    e2i, _ = get_e2i_i2e(full_entity_universe)

    df = df.loc[df["gold"].isin(set(train_entity_universe)), :].reset_index(drop=True)

    queries = sorted(df["query"])
    random.seed(VAL_SPLIT_SEED)
    random.shuffle(queries)
    queries = set(queries[:int((1 - VAL_SPLIT_PC) * len(queries))])
    df_train = df.loc[df["query"].isin(queries), :].reset_index(drop=True)
    df_val = df.loc[~df["query"].isin(queries), :].reset_index(drop=True)
    del df

    with open("translate/full_en2pl.json", "r") as f:
        en2pl = json.load(f)
    with open("translate/full_en2it.json", "r") as f:
        en2it = json.load(f)
    with open("translate/full_en2ro.json", "r") as f:
        en2ro = json.load(f)

    nia_queries, nia_gold = get_nia_queries_gold(mdl_parser, train_entity_universe)
    df_train = pd.concat([df_train, pd.DataFrame({"gold": nia_gold, "query": nia_queries, "count": [1] * len(nia_queries)})])
    df_train = add_translations(df_train, [en2pl, en2it, en2ro])
    train_queries, train_gold = expand_count_df(df_train, lambda count: count)
    val_queries, val_gold = expand_count_df(df_val, lambda count: 1)

    train_dset = cache(CACHE_DIR / "train.pkl", lambda: BiencDataset(train_queries, train_gold, quenc, mdl_cache, e2i),
                       refresh=REFRESH_CACHE)
    val_dset = cache(CACHE_DIR / "val.pkl", lambda: BiencDataset(val_queries, val_gold, quenc, mdl_cache, e2i),
                     refresh=REFRESH_CACHE)
    inf_query_dset = cache(CACHE_DIR / "inf.pkl", lambda: BiencQueryDatasetInference(val_queries, val_gold, quenc, e2i),
                           refresh=REFRESH_CACHE)

    model = Biencoder(SCORE_FN)
    model.to(device)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    inf_query_loader = DataLoader(inf_query_dset, batch_size=batch_size, shuffle=False)
    optim = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay) # TODO: Which one?

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_loader) * num_epochs) # Batch scheduler

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run = neptune.init(
        project="vmorelli/symptom-search",
        description="bienc",
        source_files=["src/**/*.py", "src/*.py"],
        tags=[experiment_name] + (["TINY"] if TINY else []) + (["DEBUG"] if DEBUG else []))

    params = {
        "experiment_name": experiment_name,
        "max_lr": max_lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimizer": optim.__class__.__name__,
        "use_amp": use_amp,
        "num_epochs": num_epochs,
        "run_id": run_start,
        "margin": margin,
        "min_count": min_count,
        "min_rel": min_rel
    }
    run["parameters"] = params

    loss_fn = BidirectionalMarginLoss(device, margin)

    # Train
    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        global_step = train_one_epoch(model, train_loader, device,
                                      loss_fn, optim, scheduler, use_amp, scaler,
                                      global_step, run)

        # Save model after every epoch
        if save_outputs:
            save_checkpoint(checkpoint_dir / f"{experiment_name}_{global_step}.pt",
                            global_step,
                            model.state_dict(),
                            optim.state_dict(),
                            scheduler.state_dict() if scheduler is not None else None,
                            scaler.state_dict() if scaler is not None else None)

        # Loss and in-batch accuracy for training validation set
        evaluate(model, val_loader, device, loss_fn, global_step, run)

    # Full evaluation on validation set
    inf_ent_dset = BiencEntityDatasetInference.from_model(model.topic_encoder,
                                                          device,
                                                          mdl_cache, train_entity_universe,
                                                          choice_method=BIENC_INFERENCE_CHOICE_METHOD)
    inf_ent_loader = DataLoader(inf_ent_dset, batch_size=batch_size, shuffle=False)
    scores, labels =  full_eval(model.content_encoder, inf_query_loader, inf_ent_loader, SCORE_FN, device)
    val_recall_dct = get_recall_dct(scores, labels)
    log_recall_dct(val_recall_dct, global_step, run, "full_val")

    # Full evaluation on test set & save artifacts
    scores, labels = eval_on_test(model, train_entity_universe, e2i, quenc, inf_ent_loader, SCORE_FN, batch_size, device)
    test_recall_dct = get_recall_dct(scores, labels)
    log_recall_dct(test_recall_dct, global_step, run, "full_test")

    # RO
    scores, labels = eval_on_ro(model, train_entity_universe, e2i, quenc, inf_ent_loader, SCORE_FN, batch_size, device)
    test_recall_dct = get_recall_dct(scores, labels)
    log_recall_dct(test_recall_dct, global_step, run, "full_test_ro")

    # Save artifacts
    Path((checkpoint_dir / f"{experiment_name}_{global_step}.pt"))
    (ARTIFACTS_DIR / f"{run_start}").mkdir(exist_ok=True, parents=True)
    torch.save(model.content_encoder, ARTIFACTS_DIR / f"{run_start}" / "query_encoder.pt")
    inf_ent_dset.to_file(ARTIFACTS_DIR / f"{run_start}" / "entity_embs.pt")

    run.stop()

if __name__ == "__main__":
    main()
