from argparse import ArgumentParser
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import bienc.tokenizer as tokenizer
from ceevee import get_source_nonsource_topics
from config import CFG
from cross.dset import CrossDataset
from cross.metrics import get_positive_class_ratio
from cross.model import CrossEncoder
from data.content import get_content2text
from data.topics import get_topic2text
from utils import flatten_positive_negative_content_ids, sanitize_fname, \
    seed_everything, get_dfs


def main():
    device = torch.device("cuda")
    CFG.NUM_WORKERS = 0
    CFG.gpus = torch.cuda.device_count()

    topics_df, content_df, corr_df = get_dfs(CFG.DATA_DIR, "cross")

    if CFG.tiny:
        corr_df = corr_df.sample(10).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(set(flatten_positive_negative_content_ids(corr_df))), :].reset_index(drop=True)
    elif CFG.small:
        corr_df = corr_df.sample(1000).reset_index(drop=True)
    elif CFG.medium:
        corr_df = corr_df.sample(10000).reset_index(drop=True)

    class_ratio = get_positive_class_ratio(corr_df)
    print(f"Positive class ratio: {class_ratio}")

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    del topics_df, content_df

    train_dset = CrossDataset(corr_df["topic_id"], corr_df["content_ids"], corr_df["cands"],
                              topic2text, content2text, CFG.CROSS_NUM_TOKENS, is_val=False)

    train_loader = DataLoader(train_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)


    model = CrossEncoder(dropout=CFG.cross_dropout)
    if CFG.gpus > 1:
        model = nn.DataParallel(model).to(device)
        print(f"Using {CFG.gpus} GPUS!")
    else:
        model = model.to(device)

    optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=CFG.use_amp)

    # Train
    global_step = 0
    for epoch in tqdm(range(CFG.num_epochs)):
        print(f"Training epoch {epoch}...")
        model.train()
        for batch in tqdm(train_loader):
            batch = tuple(x.to(device) for x in batch)
            *model_input, labels = batch
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(*model_input)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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
