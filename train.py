import random

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import neptune.new as neptune

from config import DATA_DIR, VAL_SPLIT_SEED, BIENC_MODEL_NAME, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, SCORE_FN
from data.content import get_content2text
from data.dset import BiencDataset
from data.topics import get_topic2text
from model.bienc import Biencoder
from model.losses import BidirectionalMarginLoss
from utils import get_learning_rate_momentum

TINY = True
DEBUG = False

device = torch.device("cuda") if (not DEBUG) and torch.cuda.is_available() else torch.device("cpu")

content_df = pd.read_csv(DATA_DIR / "content.csv")
corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")

if TINY:
    corr_df = corr_df.iloc[:1000, :].reset_index(drop=True)

topics_in_scope = sorted(list(set(corr_df["topic_id"])))
random.seed(VAL_SPLIT_SEED)
random.shuffle(topics_in_scope)

batch_size = 32
max_lr = 3e-5
weight_decay = 0.0
margin = 6.0
num_epochs = 2
use_amp = True
experiment_name="first"
all_folds = False


def train_one_epoch(model, train_loader, device, loss_fn, optim, scheduler, use_amp, scaler, global_step, run):
    """Train one epoch of Bi-encoder."""
    step = global_step
    model.train()
    for batch in tqdm(train_loader):
        batch = tuple(x.to(device) for x in batch)
        *model_input, entity_idxs = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            scores = model(*model_input)
            mask = torch.full_like(scores, False, dtype=torch.bool)
            mask[entity_idxs.unsqueeze(-1) == entity_idxs.unsqueeze(0)] = True
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


    train_dset = BiencDataset(train_corr_df["topic_id"], train_corr_df["content_ids"], topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, train_t2i)
    val_dset = BiencDataset(val_corr_df["topic_id"], val_corr_df["content_ids"], topic2text, content2text, TOPIC_NUM_TOKENS, CONTENT_NUM_TOKENS, val_t2i)

    model = Biencoder(SCORE_FN).to(device)
    loss_fn = BidirectionalMarginLoss(device, margin)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)


    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run = neptune.init(
        project="vmorelli/kolibri",
        source_files=["src/**/*.py", "src/*.py"],
        tags=[experiment_name] + [f"fold{fold_idx}"] + (["TINY"] if TINY else []) + (["DEBUG"] if DEBUG else []))

    # Train
    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        global_step = train_one_epoch(model, train_loader, device,
                                      loss_fn, optim, None, use_amp, scaler,
                                      global_step, run)

    fold_idx += 1
