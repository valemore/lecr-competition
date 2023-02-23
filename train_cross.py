import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import warnings

import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from neptune.common.deprecation import NeptuneDeprecationWarning
from neptune.new import Run
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import bienc.tokenizer as tokenizer
from bienc.typehints import LossFunction
from config import CFG, to_config_dct
from cross.dset import PositivesNegativesDataset
from cross.metrics import get_cross_f2, log_fscores
from cross.model import CrossEncoder
from data.content import get_content2text
from data.topics import get_topic2text
from utils import get_learning_rate_momentum, flatten_positive_negative_content_ids, get_topic_id_gold, safe_div


warnings.filterwarnings("error", category=NeptuneDeprecationWarning)


tokenizer.init_tokenizer()


def train_one_epoch(model: CrossEncoder, loss_fn: LossFunction, loader: DataLoader, device: torch.device,
                    optim: Optimizer, scheduler, use_amp: bool, scaler: GradScaler, global_step: int, run: Run) -> int:
    """Train one epoch of Cross-Encoder."""
    step = global_step
    model.train()
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(*model_input)
            loss = loss_fn(logits, labels)

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


def evaluate(model: CrossEncoder, loss_fn: LossFunction, val_loader: DataLoader, device: torch.device, num_gold: int,
             global_step: int, run: Run):
    all_probs = []
    loss_cumsum = 0.0
    num_batches = 0
    model.eval()
    for batch in tqdm(val_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        with torch.no_grad():
            logits = model(*model_input)
            loss = loss_fn(logits, labels)
        probs = logits.softmax(dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy().reshape(-1))
        loss_cumsum += loss.item()
        num_batches += 1

    all_probs = np.concatenate(all_probs)
    loss = loss_cumsum / num_batches

    print(f"Evaluation loss: {loss:.5}")
    run["cross/loss"].log(loss, step=global_step)
    return all_probs


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(CFG.output_dir)

    content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
    corr_df = pd.read_csv(CFG.CROSS_CORR_FNAME, keep_default_na=False)
    topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")

    # corr_df["cands"] = [" ".join(x.split()[:10]) for x in corr_df["cands"]]

    if CFG.tiny:
        corr_df = corr_df.sample(10).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(set(flatten_positive_negative_content_ids(corr_df))), :].reset_index(drop=True)

    class_ratio = sum(len(x.split()) for x in corr_df["content_ids"]) / sum(len(x.split()) for x in corr_df["cands"])
    print(f"Positive class ratio: {class_ratio}")

    topics_in_scope = sorted(list(set(corr_df["topic_id"])))
    random.seed(CFG.VAL_SPLIT_SEED)
    random.shuffle(topics_in_scope)

    fold_idx = 0 if CFG.folds != "no" else -1
    for topics_in_scope_train_idxs, topics_in_scope_val_idxs in KFold(n_splits=5).split(topics_in_scope):
        if (CFG.folds == "first" and fold_idx > 0) or (CFG.folds == "no" and fold_idx == 0):
            break
        if CFG.folds != "no":
            train_topics = set(topics_in_scope[idx] for idx in topics_in_scope_train_idxs)
        else:
            train_topics = topics_in_scope
        train_corr_df = corr_df.loc[corr_df["topic_id"].isin(train_topics), :].reset_index(drop=True)

        topic2text = get_topic2text(topics_df)
        content2text = get_content2text(content_df)

        train_dset = PositivesNegativesDataset(train_corr_df["topic_id"], train_corr_df["content_ids"], train_corr_df["cands"],
                                               topic2text, content2text, CFG.CROSS_NUM_TOKENS)
        train_loader = DataLoader(train_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)

        if CFG.folds != "no":
            val_topics = set(topics_in_scope[idx] for idx in topics_in_scope_val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_num_gold = sum([len(g) for g in get_topic_id_gold(val_corr_df).values()])
            val_dset = PositivesNegativesDataset(val_corr_df["topic_id"], val_corr_df["content_ids"], val_corr_df["cands"],
                                                 topic2text, content2text, CFG.CROSS_NUM_TOKENS)
            val_loader = DataLoader(val_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)

        model = CrossEncoder(dropout=CFG.cross_dropout).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
        scaler = GradScaler(enabled=CFG.use_amp)

        # Prepare logging and saving
        run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["**/*.py", "*.py"])
        run_id = f'{CFG.experiment_name}_{run["sys/id"].fetch()}'
        run["run_id"] = run_id
        run["parameters"] = to_config_dct(CFG)
        run["fold_idx"] = fold_idx
        run["part"] = "cross"
        run["run_start"] = run_start
        run["positive_class_ratio"] = class_ratio

        # Train
        global_step = 0
        for epoch in tqdm(range(CFG.num_epochs)):
            print(f"Training epoch {epoch}...")
            global_step = train_one_epoch(model, loss_fn, train_loader, device, optim, None, CFG.use_amp, scaler,
                                          global_step, run)

            if CFG.folds != "no":
                # Loss and in-batch accuracy for training validation set
                print(f"Evaluating epoch {epoch}...")
                all_probs = evaluate(model, loss_fn, val_loader, device, val_num_gold, global_step, run)
                fscores = get_cross_f2(all_probs, val_corr_df)
                del all_probs
                log_fscores(fscores, global_step, run)
                del fscores

        # Save artifacts
        out_dir = output_dir / f"{run_id}" / "cross"
        out_dir.mkdir(parents=True, exist_ok=True)
        # (output_dir / f"{run_id}" / "tokenizer").mkdir(parents=True, exist_ok=True)
        model.save(out_dir)
        # tokenizer.tokenizer.save_pretrained(output_dir / f"{run_id}" / "tokenizer")

        fold_idx += 1
        run.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--df", type=str, required=True)
    parser.add_argument("--cross_dropout", default=0.1, type=float)
    parser.add_argument("--folds", type=str, choices=["first", "all", "no"], default="first")
    parser.add_argument("--output_dir", type=str, default="../out")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--val_split_seed", type=int)
    parser.add_argument("--cross_model_name", type=str)
    parser.add_argument("--cross_num_tokens", type=int)
    parser.add_argument("--num_workers", type=int)

    args = parser.parse_args()

    CFG.tiny = args.tiny
    CFG.batch_size = args.batch_size
    CFG.max_lr = args.max_lr
    CFG.weight_decay = args.weight_decay
    CFG.num_epochs = args.num_epochs
    CFG.use_amp = not args.use_fp
    CFG.experiment_name = args.experiment_name
    CFG.CROSS_CORR_FNAME = args.df
    CFG.cross_dropout = args.cross_dropout
    CFG.folds = args.folds
    CFG.output_dir = args.output_dir

    if args.data_dir is not None:
        CFG.DATA_DIR = Path(args.data_dir)
    if args.val_split_seed is not None:
        CFG.VAL_SPLIT_SEED = args.val_split_seed
    if args.cross_model_name is not None:
        CFG.CROSS_MODEL_NAME = args.cross_model_name
    if args.cross_num_tokens is not None:
        CFG.CROSS_NUM_TOKENS = args.cross_num_tokens
    if args.num_workers is not None:
        CFG.NUM_WORKERS = args.num_workers

    main()
