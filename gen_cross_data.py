from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd

from bienc.inference import embed_and_nn, entities_inference
from bienc.tokenizer import init_tokenizer
from config import CFG
from data.content import get_content2text
from data.topics import get_topic2text
from submit import get_biencoder, get_test_topic_ids
from utils import get_topic_id_gold


if __name__ == "__main__":
    device = torch.device("cuda")

    parser = ArgumentParser()
    parser.add_argument("--bienc_path", required=True, type=str)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--num_neighbors", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="../data/cross_corr.csv")

    args = parser.parse_args()

    if args.num_neighbors is not None:
        CFG.NUM_NEIGHBORS = args.num_neighbors
    CFG.batch_size = args.batch_size
    out_fname = Path(args.out)

    bienc = get_biencoder(args.bienc_path, device)
    init_tokenizer(args.tokenizer_path)
    content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
    topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")
    corr_df = pd.read_csv(CFG.DATA_DIR / "correlations.csv")
    content_ids = sorted(list(set(content_df["id"])))
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}
    i2c = {idx: content_id for content_id, idx in c2i.items()}

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    # corr_df = corr_df.iloc[:20, :]
    # content_ids = sorted(list(set(content_ids[:200]) | set([y for x in corr_df.loc[:, "content_ids"].tolist() for y in x.split()])))
    topic_ids = corr_df["topic_id"]

    nn_model = embed_and_nn(bienc, content_ids, content2text, CFG.NUM_NEIGHBORS, CFG.batch_size, device)
    distances, indices = entities_inference(topic_ids, bienc, nn_model, topic2text, device, CFG.batch_size)

    t2gold = get_topic_id_gold(corr_df)

    negative_ids = []
    for topic_id, pred_idxs in zip(topic_ids, indices):
        gold = t2gold[topic_id]
        # gold_idxs = [c2i[content_id] for content_id in t2gold[topic_id]]
        negatives = [i2c[pred_idx] for pred_idx in pred_idxs if i2c[pred_idx] not in gold]
        negative_ids.append(" ".join(negatives))
    gen_df = corr_df.copy()
    gen_df["negative_cands"] = negative_ids

    gen_df.to_csv(out_fname, index=False)
