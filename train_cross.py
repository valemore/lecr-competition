import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import neptune.new as neptune
import torch
import torch.nn as nn
import torch.nn.functional as F
from neptune.new import Run
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import cross.tokenizer as tokenizer
from ceevee import get_source_nonsource_topics, filter_duplicates_inplace
from config import CFG, to_config_dct
from cross.dset import CrossDataset
from cross.metrics import get_positive_class_ratio, get_cross_f2, log_fscores
from cross.model import CrossEncoder
from cross.post import post_process
from data.content import get_content2text
from data.topics import get_topic2text
from utils import flatten_positive_negative_content_ids, sanitize_fname, \
    seed_everything, get_dfs, get_learning_rate_momentum, save_checkpoint


def train_one_epoch(model: CrossEncoder, loader: DataLoader, device: torch.device,
                    optim: Optimizer, scheduler, use_amp: bool, scaler: GradScaler, global_step: int, run: Run) -> int:
    """Train one epoch of Cross-Encoder."""
    step = global_step
    model.train()
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(*model_input)
            loss = F.cross_entropy(logits, labels)

        optim.zero_grad()
        scaler.scale(loss).backward()
        if CFG.clip:
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.clip)
        scaler.step(optim)
        scaler.update()

        if CFG.scheduler == "cosine":
            scheduler.step()

        # Log
        run["loss"].log(loss.item(), step=step)
        lr, momentum = get_learning_rate_momentum(optim)
        run["lr"].log(lr, step=step)
        if momentum:
            run["momentum"].log(momentum, step=step)

        step += 1
    return step


def evaluate(model: CrossEncoder, dset: CrossDataset, device: torch.device,
             global_step: int, run: Run):
    loader = DataLoader(dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    all_probs = torch.empty(len(dset), dtype=torch.float)
    loss_cumsum = 0.0
    num_batches = 0
    model.eval()
    i = 0
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, labels = batch
        # TODO Autocast?
        with torch.no_grad():
            logits = model(*model_input)
        loss = F.cross_entropy(logits, labels)
        probs = logits.softmax(dim=1)[:, 1]
        bs = logits.shape[0]
        all_probs[i:(i+bs)] = probs.cpu().reshape(-1)
        loss_cumsum += loss.item()
        i += bs
        num_batches += 1

    loss = loss_cumsum / num_batches

    print(f"Evaluation loss: {loss:.5}")
    run["cross/loss"].log(loss, step=global_step)
    return all_probs.numpy(), loss


def main():
    device = torch.device("cuda")
    CFG.gpus = torch.cuda.device_count()
    if CFG.gpus > 1:
        CFG.NUM_WORKERS = 0
    output_dir = Path(CFG.output_dir)
    checkpoint_dir = Path(CFG.checkpoint_dir)

    topics_df, content_df, corr_df = get_dfs(CFG.DATA_DIR, "cross")

    corr_df = filter_duplicates_inplace(corr_df, topics_df, CFG.DUP_FILTER_DEPTH)

    if CFG.tiny:
        corr_df = corr_df.sample(10).sort_values("topic_id").reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(set(flatten_positive_negative_content_ids(corr_df))), :].reset_index(drop=True)
    elif CFG.small:
        corr_df = corr_df.sample(1000).reset_index(drop=True)
    elif CFG.medium:
        corr_df = corr_df.sample(10000).reset_index(drop=True)

    class_ratio = get_positive_class_ratio(corr_df)
    print(f"Positive class ratio: {class_ratio}")

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    source_topics, nonsource_topics = get_source_nonsource_topics(corr_df, topics_df)

    del topics_df, content_df

    fold_idx = 0 if CFG.folds != "no" else -1
    for train_idxs, val_idxs in KFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.VAL_SPLIT_SEED).split(nonsource_topics):
        if (CFG.folds == "first" and fold_idx > 0) or (CFG.folds == "no" and fold_idx == 0):
            break
        print(f"---*** Training fold {fold_idx} ***---")
        if CFG.folds != "no":
            train_topics = set(nonsource_topics[idx] for idx in train_idxs) | set(source_topics)
        else:
            train_topics = set(corr_df["topic_id"])
        train_corr_df = corr_df.loc[corr_df["topic_id"].isin(train_topics), :].reset_index(drop=True)

        train_dset = CrossDataset(train_corr_df["topic_id"], train_corr_df["content_ids"], train_corr_df["cands"],
                                  topic2text, content2text, CFG.CROSS_NUM_TOKENS, is_val=False)

        train_loader = DataLoader(train_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)

        if CFG.folds != "no":
            val_topics = set(nonsource_topics[idx] for idx in val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_dset = CrossDataset(val_corr_df["topic_id"], val_corr_df["content_ids"], val_corr_df["cands"],
                                    topic2text, content2text, CFG.CROSS_NUM_TOKENS, is_val=True)

        model = CrossEncoder(dropout=CFG.cross_dropout)
        if CFG.gpus > 1:
            model = nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
        scaler = GradScaler(enabled=CFG.use_amp)
        if CFG.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optim, mode="max", patience=2)
        elif CFG.scheduler == "cosine":
            scheduler = CosineAnnealingWarmRestarts(optim, T_0=CFG.num_epochs * len(train_loader))
        else:
            scheduler = None

        # Prepare logging and saving
        run_start = datetime.utcnow().strftime("%m%d-%H%M%S")
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["**/*.py", "*.py"])
        run["cmd"] = " ".join(sys.argv)
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
            global_step = train_one_epoch(model, train_loader, device, optim, scheduler, CFG.use_amp, scaler,
                                          global_step, run)

            if CFG.folds != "no":
                # Loss and in-batch accuracy for training validation set
                print(f"Evaluating epoch {epoch}...")
                all_probs, loss = evaluate(model, val_dset, device, global_step, run)
                all_probs = post_process(all_probs, val_dset.topic_ids)
                fscores = get_cross_f2(all_probs, val_corr_df)
                del all_probs
                log_fscores(fscores, global_step, run)
                del fscores

                if CFG.scheduler == "plateau":
                    scheduler.step(loss)

            # Save checkpoint
            if CFG.checkpoint:
                if CFG.gpus > 1:
                    save_checkpoint(checkpoint_dir / f"{run_id}" / f"epoch-{epoch}.pt", global_step,
                                    model.module.state_dict(), optim.state_dict(), None, scaler.state_dict())
                else:
                    save_checkpoint(checkpoint_dir / f"{run_id}" / f"epoch-{epoch}.pt", global_step,
                                    model.state_dict(), optim.state_dict(), None, scaler.state_dict())

        # Save artifacts
        out_dir = output_dir / f"{run_id}" / "cross"
        out_dir.mkdir(parents=True, exist_ok=True)
        # (output_dir / f"{run_id}" / "tokenizer").mkdir(parents=True, exist_ok=True)
        if CFG.gpus > 1:
            model.module.save(out_dir)
        else:
            model.save(out_dir)
        # tokenizer.tokenizer.save_pretrained(output_dir / f"{run_id}" / "tokenizer")

        fold_idx += 1
        run.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--df", type=str, required=True)
    parser.add_argument("--cross_dropout", default=0.1, type=float)
    parser.add_argument("--cross_num_cands", required=True, type=int)
    parser.add_argument("--folds", type=str, choices=["first", "all", "no"], default="first")
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="../cout")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="../check-cross")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--val_split_seed", type=int)
    parser.add_argument("--training_seed", type=int)
    parser.add_argument("--cross_model_name", type=str)
    parser.add_argument("--cross_num_tokens", type=int)
    parser.add_argument("--num_workers", type=int)

    args = parser.parse_args()

    CFG.tiny = args.tiny
    CFG.medium = args.medium
    CFG.small = args.small
    CFG.batch_size = args.batch_size
    CFG.max_lr = args.max_lr
    CFG.weight_decay = args.weight_decay
    CFG.clip = args.clip
    CFG.num_epochs = args.num_epochs
    CFG.use_amp = not args.use_fp
    CFG.scheduler = args.scheduler
    CFG.experiment_name = sanitize_fname(args.experiment_name)
    CFG.cross_corr_fname = args.df
    CFG.cross_dropout = args.cross_dropout
    CFG.cross_num_cands = args.cross_num_cands
    CFG.folds = args.folds
    CFG.num_folds = args.num_folds
    CFG.output_dir = args.output_dir
    CFG.checkpoint = args.checkpoint
    CFG.checkpoint_dir = args.checkpoint_dir

    if args.data_dir is not None:
        CFG.DATA_DIR = Path(args.data_dir)
    if args.val_split_seed is not None:
        CFG.VAL_SPLIT_SEED = args.val_split_seed
    if args.training_seed is not None:
        CFG.TRAINING_SEED = args.training_seed
    if args.cross_model_name is not None:
        CFG.CROSS_MODEL_NAME = args.cross_model_name
    if args.cross_num_tokens is not None:
        CFG.CROSS_NUM_TOKENS = args.cross_num_tokens
    if args.num_workers is not None:
        CFG.NUM_WORKERS = args.num_workers

    seed_everything(CFG.TRAINING_SEED)
    tokenizer.init_tokenizer()
    torch.set_float32_matmul_precision("medium") # TODO?

    main()
