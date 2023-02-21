import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict
import warnings

import neptune.new as neptune
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
from cross.model import CrossEncoder
from data.content import get_content2text
from data.topics import get_topic2text
from utils import get_learning_rate_momentum, flatten_positive_negative_content_ids


warnings.filterwarnings("error", category=NeptuneDeprecationWarning)


tokenizer.init_tokenizer()


def train_one_epoch(model: CrossEncoder, loss_fn: LossFunction, loader: DataLoader, device: torch.device,
                    optim: Optimizer, scheduler, use_amp: bool, scaler: GradScaler, global_step: int, run: Run) -> int:
    """Train one epoch of Bi-encoder."""
    step = global_step
    model.train()
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            scores = model(*model_input)
            loss = loss_fn(scores, labels.reshape(-1, 1).float())

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


def evaluate(model: CrossEncoder, loss_fn: LossFunction, val_loader: DataLoader, device: torch.device, global_step: int,
             run: Run) -> Dict[str, float]:
    """Performs in-batch validation."""
    acc_cumsum = 0.0
    loss_cumsum = 0.0
    num_examples = 0
    num_batches = 0

    model.eval()
    for batch in tqdm(val_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        with torch.no_grad():
            scores = model(*model_input)
            loss = loss_fn(scores, labels.reshape(-1, 1).float())
        bs = scores.shape[0]
        preds = (scores >= 0.0).int().reshape(-1)

        acc_cumsum += (preds == labels).sum().item()
        loss_cumsum += loss.item()
        num_examples += bs
        num_batches += 1

    acc = acc_cumsum / num_examples
    loss = loss_cumsum / num_batches

    print(f"Evaluation accuracy: {acc:.5}")
    print(f"Evaluation loss: {loss:.5}")

    run["val/acc"].log(acc, step=global_step)
    run["val/loss"].log(loss, step=global_step)

    return {"acc": acc, "loss": loss}


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(CFG.output_dir)

    content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
    corr_df = pd.read_csv(CFG.DATA_DIR / CFG.CROSS_CORR_FNAME)
    topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")

    if CFG.tiny:
        corr_df = corr_df.sample(10).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(
            set(flatten_positive_negative_content_ids(corr_df)) | set(content_df["id"].sample(1000))), :].reset_index(drop=True)

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

        train_dset = PositivesNegativesDataset(train_corr_df["topic_id"], train_corr_df["content_ids"], train_corr_df["negative_cands"],
                                               topic2text, content2text, CFG.CROSS_NUM_TOKENS)
        train_loader = DataLoader(train_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)

        if CFG.folds != "no":
            val_topics = set(topics_in_scope[idx] for idx in topics_in_scope_val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_dset = PositivesNegativesDataset(val_corr_df["topic_id"], val_corr_df["content_ids"], val_corr_df["negative_cands"],
                                                 topic2text, content2text, CFG.CROSS_NUM_TOKENS)
            val_loader = DataLoader(val_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)

        model = CrossEncoder().to(device)
        loss_fn = nn.BCEWithLogitsLoss().to(device)

        optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
        scaler = GradScaler(enabled=CFG.use_amp)

        # Prepare logging and saving
        run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["**/*.py", "*.py"])
        run["parameters"] = to_config_dct(CFG)
        run["fold_idx"] = fold_idx

        # Train
        global_step = 0
        for epoch in tqdm(range(CFG.num_epochs)):
            print(f"Training epoch {epoch}...")
            global_step = train_one_epoch(model, loss_fn, train_loader, device, optim, None, CFG.use_amp, scaler,
                                          global_step, run)

            if CFG.folds != "no":
                # Loss and in-batch accuracy for training validation set
                print(f"Evaluating epoch {epoch}...")
                evaluate(model, loss_fn, val_loader, device, global_step, run)

                # Evaluate inference
                # TODO

        # Save artifacts
        (output_dir / f"{CFG.experiment_name}_{run_start}" / "cross").mkdir(parents=True, exist_ok=True)
        # (output_dir / f"{CFG.experiment_name}_{run_start}" / "tokenizer").mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(output_dir / f"{CFG.experiment_name}_{run_start}" / "cross")
        # tokenizer.tokenizer.save_pretrained(output_dir / f"{CFG.experiment_name}_{run_start}" / "tokenizer")

        fold_idx += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--experiment_name", type=str, required=True)
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
