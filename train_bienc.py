import gc
from argparse import ArgumentParser
from pathlib import Path
import random
from typing import Dict, Tuple, Union

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

from bienc.gen_cross import gen_cross_df
from config import CFG, to_config_dct
from bienc.inference import embed_and_nn, entities_inference, predict_entities
import bienc.tokenizer as tokenizer
from bienc.typehints import LossFunction
from data.content import get_content2text
from bienc.dset import BiencDataset
from data.topics import get_topic2text
from bienc.model import Biencoder, BiencoderModule
from bienc.losses import BidirectionalMarginLoss
from metrics import get_fscore
from utils import get_learning_rate_momentum, flatten_content_ids, are_entity_ids_aligned, get_topic_id_gold, \
    sanitize_model_name
from bienc.metrics import get_bienc_thresh_metrics, log_dct, BIENC_STANDALONE_THRESHS, get_log_mir_metrics, \
    get_bienc_cands_metrics, get_average_precision_cands, get_avg_precision_threshs

tokenizer.init_tokenizer()


def train_one_epoch(model: Biencoder, loss_fn: LossFunction, train_loader: DataLoader, device: torch.device,
                    optim: Optimizer, scheduler, use_amp: bool, scaler: GradScaler, global_step: int, run: Run) -> int:
    """Train one epoch of Bi-encoder."""
    step = global_step
    model.train()
    for batch in tqdm(train_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, topic_idxs, content_idxs = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            scores = model(*model_input)
            mask = torch.full_like(scores, False, dtype=torch.bool)
            mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
            mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
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
        *model_input, topic_idxs, content_idxs = batch
        with torch.no_grad():
            scores = model(*model_input)
            mask = torch.full_like(scores, False, dtype=torch.bool)
            mask[topic_idxs.unsqueeze(-1) == topic_idxs.unsqueeze(0)] = True
            mask[content_idxs.unsqueeze(-1) == content_idxs.unsqueeze(0)] = True
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

    # Metrics
    t2gold = get_topic_id_gold(corr_df)

    # Thresh metrics
    precision_dct, recall_dct, micro_prec_dct, pcr_dct = get_bienc_thresh_metrics(distances, indices, data_ids, e2i, t2gold)
    avg_precision = get_avg_precision_threshs(distances, indices, data_ids, e2i, t2gold)
    print(f"Mean average precision @ {CFG.NUM_NEIGHBORS}: {avg_precision:.5}")
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
                            gen_cross: bool,
                            global_step: int, run: Run) -> Tuple[Optimizer, Union[None, pd.DataFrame]]:
    optimizer_state_dict = optim.state_dict()
    cross_df = evaluate_inference(model.topic_encoder, device, batch_size,
                                  corr_df, topic2text, content2text, e2i,
                                  gen_cross,
                                  global_step, run)
    optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
    optim.load_state_dict(optimizer_state_dict)
    return optim, cross_df


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(CFG.output_dir)

    content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
    corr_df = pd.read_csv(CFG.DATA_DIR / "correlations.csv")
    topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")

    if CFG.tiny:
        corr_df = corr_df.sample(1000).reset_index(drop=True)
        content_df = content_df.loc[content_df["id"].isin(
            set(flatten_content_ids(corr_df)) | set(content_df["id"].sample(1000))), :].reset_index(drop=True)

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
        train_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(train_corr_df["topic_id"]))))}

        c2i = {content_id: content_idx for content_idx, content_id in enumerate(sorted(set(content_df["id"])))}
        topic2text = get_topic2text(topics_df)
        content2text = get_content2text(content_df)
        del topics_df, content_df

        train_dset = BiencDataset(train_corr_df["topic_id"], train_corr_df["content_ids"],
                                  topic2text, content2text, CFG.TOPIC_NUM_TOKENS, CFG.CONTENT_NUM_TOKENS, train_t2i, c2i)
        train_loader = DataLoader(train_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=True)

        if CFG.folds != "no":
            val_topics = set(topics_in_scope[idx] for idx in topics_in_scope_val_idxs)
            val_corr_df = corr_df.loc[corr_df["topic_id"].isin(val_topics), :].reset_index(drop=True)
            val_t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(val_corr_df["topic_id"]))))}
            val_dset = BiencDataset(val_corr_df["topic_id"], val_corr_df["content_ids"],
                                    topic2text, content2text, CFG.TOPIC_NUM_TOKENS, CFG.CONTENT_NUM_TOKENS, val_t2i, c2i)
            val_loader = DataLoader(val_dset, batch_size=CFG.batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)

        model = Biencoder().to(device)
        loss_fn = BidirectionalMarginLoss(device, CFG.margin)

        optim = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)
        scaler = GradScaler(enabled=not CFG.use_fp)

        # Prepare logging and saving
        run = neptune.init_run(
            project="vmorelli/kolibri",
            source_files=["**/*.py", "*.py"])
        run_id = f'{CFG.experiment_name}_{run["sys/id"].fetch()}'
        run["run_id"] = run_id
        run["parameters"] = to_config_dct(CFG)
        run["fold_idx"] = fold_idx
        run["part"] = "bienc"

        # Train
        global_step = 0
        for epoch in tqdm(range(CFG.num_epochs)):
            print(f"Training epoch {epoch}...")
            gc.collect()
            torch.cuda.empty_cache()
            global_step = train_one_epoch(model, loss_fn, train_loader, device, optim, None, not CFG.use_fp, scaler,
                                          global_step, run)

            if CFG.folds != "no":
                # Loss and in-batch accuracy for training validation set
                print(f"Running in-batch evaluation for epoch {epoch}...")
                evaluate(model, loss_fn, val_loader, device, global_step, run)

                # Evaluate inference
                print(f"Running inference-mode evaluation for epoch {epoch}...")
                # We need to re-initialize optimizer because evaluate_inference offloads model onto CPU
                # Keep for safety. Tests indicate this is not actually needed, but no guarantee from docs.
                optim, cross_df = wrap_evaluate_inference(model, device, CFG.batch_size,
                                                          val_corr_df, topic2text, content2text, c2i,
                                                          optim,
                                                          epoch == CFG.num_epochs - 1,
                                                          global_step, run)
                if epoch == CFG.num_epochs - 1:
                    (output_dir / f"{run_id}" / "cross").mkdir(parents=True, exist_ok=True)
                    cross_df_fname = output_dir / f"{run_id}" / "cross" / f"{run_id}_fold-{fold_idx}.csv"
                    cross_df.to_csv(cross_df_fname, index=False)


        # Save artifacts
        (output_dir / f"{run_id}" / "bienc").mkdir(parents=True, exist_ok=True)
        (output_dir / f"{run_id}" / "tokenizer").mkdir(parents=True, exist_ok=True)
        model.content_encoder.encoder.save_pretrained(output_dir / f"{run_id}" / "bienc")
        tokenizer.tokenizer.save_pretrained(output_dir / f"{run_id}" / "tokenizer")

        fold_idx += 1
        run.stop()

    return {"run_id": run_id, "CFG": to_config_dct(CFG)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--use_fp", action="store_true")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--folds", type=str, choices=["first", "all", "no"], default="first")
    parser.add_argument("--output_dir", type=str, default="../out")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--val_split_seed", type=int)
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
    CFG.use_fp = args.use_fp
    CFG.experiment_name = sanitize_model_name(args.experiment_name)
    CFG.folds = args.folds
    CFG.output_dir = args.output_dir

    if args.data_dir is not None:
        CFG.DATA_DIR = Path(args.data_dir)
    if args.val_split_seed is not None:
        CFG.VAL_SPLIT_SEED = args.val_split_seed
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

    main()
