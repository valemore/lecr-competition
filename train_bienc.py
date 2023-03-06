import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import neptune.new as neptune
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import bienc.tokenizer as tokenizer
from bienc.dset import BiencDataset
from bienc.losses import BidirectionalMarginLoss
from bienc.model import Biencoder
from bienc.sampler import SameLanguageSampler
from ceevee import get_topics_in_corr, filter_duplicates_inplace
from config import CFG, to_config_dct
from data.content import get_content2text
from data.topics import get_topic2text
from bienc.trainer import LitBienc
from ignorewarnings import IGNORE_LIST
from utils import flatten_content_ids, sanitize_fname, get_t2lang_c2lang, seed_everything, get_dfs, \
    get_content_ids_c2i

for warning_msg in IGNORE_LIST:
    warnings.filterwarnings("ignore", message=warning_msg)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(CFG.output_dir)
    cross_output_dir = Path(CFG.cross_output_dir)

    topics_df, content_df, corr_df = get_dfs(CFG.DATA_DIR, "bienc")

    corr_df = filter_duplicates_inplace(corr_df, topics_df, CFG.DUP_FILTER_DEPTH)

    if CFG.tiny:
        corr_df = corr_df.sample(1000).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(
            set(flatten_content_ids(corr_df)) | set(content_df["id"].sample(1000))), :].reset_index(drop=True)

    t2lang, c2lang = get_t2lang_c2lang(corr_df, content_df)
    topics_in_corr = get_topics_in_corr(corr_df)

    _, c2i = get_content_ids_c2i(content_df)
    if CFG.FILTER_LANG:
        c2i["dummy"] = -1
    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    del topics_df, content_df

    experiment_id = f'{CFG.experiment_name}_{datetime.utcnow().strftime("%m%d-%H%M%S")}'

    fold_idx = 0 if CFG.folds != "no" else -1
    for train_idxs, val_idxs in KFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.VAL_SPLIT_SEED).split(topics_in_corr):
        if (CFG.folds == "first" and fold_idx > 0) or (CFG.folds == "no" and fold_idx == 0):
            break
        print(f"---*** Training fold {fold_idx} ***---")
        if CFG.folds != "no":
            train_topics = set(topics_in_corr[idx] for idx in train_idxs)
        else:
            train_topics = topics_in_corr
        train_corr_df = corr_df.loc[corr_df["topic_id"].isin(train_topics), :].reset_index(drop=True)
        train_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(train_corr_df["topic_id"]))))}

        train_dset = BiencDataset(train_corr_df["topic_id"], train_corr_df["content_ids"],
                                  train_corr_df["language"],
                                  topic2text, content2text, CFG.TOPIC_NUM_TOKENS, CFG.CONTENT_NUM_TOKENS, train_t2i, c2i)

        def get_train_loader(batch_size):
            return DataLoader(train_dset, num_workers=CFG.NUM_WORKERS,
                              batch_sampler=SameLanguageSampler(train_dset, batch_size))

        if CFG.folds != "no":
            val_topics = set(topics_in_corr[idx] for idx in val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(val_corr_df["topic_id"]))))}
            val_dset = BiencDataset(val_corr_df["topic_id"], val_corr_df["content_ids"],
                                    val_corr_df["language"],
                                    topic2text, content2text, CFG.TOPIC_NUM_TOKENS, CFG.CONTENT_NUM_TOKENS, val_t2i, c2i)
            val_loader = DataLoader(val_dset, num_workers=CFG.NUM_WORKERS,
                                    batch_sampler=SameLanguageSampler(val_dset, CFG.batch_size))
        else:
            val_loader = None

        model = Biencoder().to(device)
        loss_fn = BidirectionalMarginLoss(device, CFG.margin)

        # Prepare logging and saving
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["**/*.py", "*.py"])
        run["cmd"] = " ".join(sys.argv)
        run_id = f'{experiment_id}_{run["sys/id"].fetch()}'
        run["run_id"] = run_id
        run["experiment_id"] = experiment_id
        run["parameters"] = to_config_dct(CFG)
        run["fold_idx"] = fold_idx
        run["part"] = "bienc"

        lit_model = LitBienc(model, loss_fn, get_train_loader,
                             topic2text, content2text, c2i, t2lang, c2lang,
                             CFG.max_lr, CFG.weight_decay, CFG.batch_size,
                             cross_output_dir, experiment_id,
                             CFG.folds, fold_idx, val_corr_df if CFG.folds != "no" else None,
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
            (output_dir / f"{run_id}" / "bienc").mkdir(parents=True, exist_ok=True)
            (output_dir / f"{run_id}" / "tokenizer").mkdir(parents=True, exist_ok=True)
            model.content_encoder.encoder.save_pretrained(output_dir / f"{run_id}" / "bienc")
            tokenizer.tokenizer.save_pretrained(output_dir / f"{run_id}" / "tokenizer")
            print(f'Saved model artifacts to {str(output_dir / f"{run_id}")}')

        fold_idx += 1
        run.stop()

    if CFG.folds == "all":
        cross_df = pd.DataFrame()
        for fold_idx in range(CFG.num_folds):
            cross_df = pd.concat([cross_df, pd.read_csv(cross_output_dir / f"{experiment_id}" / f"fold-{fold_idx}.csv", keep_default_na=False)]).reset_index(drop=True)
        cross_df = cross_df.sort_values("topic_id").reset_index(drop=True)
        cross_df.to_csv(cross_output_dir / f"{experiment_id}" / "all_folds.csv", index=False)
        print(f'Wrote cross df to {cross_output_dir / f"{experiment_id}" / "all_folds.csv"}')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--tune_lr", action="store_true")
    parser.add_argument("--tune_bs", action="store_true")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--folds", type=str, choices=["first", "all", "no"], default="first")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="../out")
    parser.add_argument("--cross_output_dir", type=str, default="../cross")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--val_split_seed", type=int)
    parser.add_argument("--training_seed", type=int)
    parser.add_argument("--bienc_model_name", type=str)
    parser.add_argument("--topic_num_tokens", type=int)
    parser.add_argument("--content_num_tokens", type=int)
    parser.add_argument("--score_fn",choices=["cos_sim", "dot_score", "l2_dot_score"], type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_neighbors", type=int)

    args = parser.parse_args()

    CFG.tiny = args.tiny
    CFG.batch_size = args.batch_size
    CFG.max_lr = args.max_lr
    CFG.weight_decay = args.weight_decay
    CFG.margin = args.margin
    CFG.num_epochs = args.num_epochs
    CFG.use_amp = not args.use_fp
    CFG.tune_lr = args.tune_lr
    CFG.tune_bs = args.tune_bs
    CFG.scheduler = args.scheduler
    CFG.experiment_name = sanitize_fname(args.experiment_name)
    CFG.folds = args.folds
    CFG.num_folds = args.num_folds
    CFG.output_dir = args.output_dir
    CFG.cross_output_dir = args.cross_output_dir

    if args.data_dir is not None:
        CFG.DATA_DIR = Path(args.data_dir)
    if args.val_split_seed is not None:
        CFG.VAL_SPLIT_SEED = args.val_split_seed
    if args.training_seed is not None:
        CFG.TRAINING_SEED = args.training_seed
    if args.bienc_model_name is not None:
        CFG.BIENC_MODEL_NAME = args.bienc_model_name
    if args.topic_num_tokens is not None:
        CFG.TOPIC_NUM_TOKENS = args.topic_num_tokens
    if args.content_num_tokens is not None:
        CFG.CONTENT_NUM_TOKENS = args.content_num_tokens
    if args.score_fn is not None:
        CFG.SCORE_FN = args.score_fn
    if args.num_workers is not None:
        CFG.NUM_WORKERS = args.num_workers
    if args.num_neighbors is not None:
        CFG.NUM_NEIGHBORS = args.num_neighbors

    assert not (CFG.tune_lr and CFG.tune_bs), "Can't tune both at the same time without breaking logging."

    seed_everything(CFG.TRAINING_SEED)
    tokenizer.init_tokenizer()
    torch.set_float32_matmul_precision("medium") # TODO?

    main()
