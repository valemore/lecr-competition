from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import neptune.new as neptune
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import bienc.tokenizer as tokenizer
from ceevee import get_source_nonsource_topics
from config import CFG, to_config_dct
from cross.dset import CrossDataset
from cross.metrics import get_positive_class_ratio
from cross.model import CrossEncoder
from cross.trainer import LitCross
from data.content import get_content2text
from data.topics import get_topic2text
from utils import flatten_positive_negative_content_ids, sanitize_fname, \
    seed_everything, get_dfs


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(CFG.output_dir)
    checkpoint_dir = Path(CFG.checkpoint_dir)

    topics_df, content_df, corr_df = get_dfs(CFG.DATA_DIR, "cross")

    if CFG.tiny:
        corr_df = corr_df.sample(10).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(set(flatten_positive_negative_content_ids(corr_df))), :].reset_index(drop=True)

    class_ratio = get_positive_class_ratio(corr_df, CFG.cross_num_cands)
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
                                  topic2text, content2text, CFG.CROSS_NUM_TOKENS, num_cands=CFG.cross_num_cands, is_val=False)

        def get_train_loader(batch_size):
            return DataLoader(train_dset, batch_size=batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)

        if CFG.folds != "no":
            val_topics = set(nonsource_topics[idx] for idx in val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_dset = CrossDataset(val_corr_df["topic_id"], val_corr_df["content_ids"], val_corr_df["cands"],
                                    topic2text, content2text, CFG.CROSS_NUM_TOKENS, CFG.cross_num_cands, is_val=True)
            val_loader = DataLoader(val_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
        else:
            val_corr_df = None
            val_loader = None

        model = CrossEncoder(dropout=CFG.cross_dropout).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

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

        lit_model = LitCross(model, loss_fn, get_train_loader,
                             val_corr_df,
                             CFG.max_lr, CFG.weight_decay, CFG.batch_size,
                             run)
        trainer = pl.Trainer(accelerator="gpu", devices=1,
                             max_epochs=CFG.num_epochs,
                             precision=16 if CFG.use_amp else 32,
                             logger=False,
                             enable_checkpointing=False,
                             auto_lr_find=CFG.tune_lr, auto_scale_batch_size=CFG.tune_bs)
        if CFG.tune_bs:
            trainer.tune(model=lit_model, scale_batch_size_kwargs={"mode": "binsearch"})
            run["tuned_bs"] = lit_model.batch_size
            return
        if CFG.tune_lr:
            trainer.tune(model=lit_model)
            run["tuned_lr"] = lit_model.learning_rate
            return
        trainer.fit(model=lit_model, val_dataloaders=val_loader)

        # Save artifacts
        if not (CFG.tune_bs or CFG.tune_lr):
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--tune_lr", action="store_true")
    parser.add_argument("--tune_bs", action="store_true")
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
    CFG.batch_size = args.batch_size
    CFG.max_lr = args.max_lr
    CFG.weight_decay = args.weight_decay
    CFG.num_epochs = args.num_epochs
    CFG.use_amp = not args.use_fp
    CFG.tune_lr = args.tune_lr
    CFG.tune_bs = args.tune_bs
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

    assert not (CFG.tune_lr and CFG.tune_bs), "Can't tune both at the same time without breaking logging."

    seed_everything(CFG.TRAINING_SEED)
    tokenizer.init_tokenizer()
    torch.set_float32_matmul_precision("medium") # TODO?

    main()
