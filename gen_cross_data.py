from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
import neptune.new as neptune

from bienc.inference import embed_and_nn, entities_inference
from bienc.tokenizer import init_tokenizer
from config import CFG
from data.content import get_content2text
from data.topics import get_topic2text
from submit import get_bienc
from utils import get_topic_id_gold


if __name__ == "__main__":
    raise Exception("Obsolete")
    device = torch.device("cuda")

    parser = ArgumentParser()
    parser.add_argument("--bienc_path", required=True, type=str)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--num_neighbors", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--out", type=str, default="../data/cross_corr.csv")

    args = parser.parse_args()

    if args.num_neighbors is not None:
        CFG.NUM_NEIGHBORS = args.num_neighbors
    bienc_run_id = Path(args.bienc_path).parent.name
    if args.out is None:
        cross_dir = Path("../cross")
        cross_dir.mkdir(exist_ok=True, parents=True)
        args.out = str(cross_dir / f"{bienc_run_id}.csv")
    out_fname = Path(args.out)
    CFG.batch_size = args.batch_size

    bienc = get_bienc(args.bienc_path, device)
    init_tokenizer(args.tokenizer_path)
    content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
    topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")
    corr_df = pd.read_csv(CFG.DATA_DIR / "correlations.csv")
    content_ids = sorted(list(set(content_df["id"])))
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}
    i2c = {idx: content_id for content_id, idx in c2i.items()}

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)
    topic_ids = corr_df["topic_id"]

    print(f"Generating Cross-Encoder training data from {args.bienc_path} and {args.tokenizer_path}...")
    print(f"Using {CFG.NUM_NEIGHBORS} neigbors...")

    nn_model = embed_and_nn(bienc, content_ids, content2text, CFG.NUM_NEIGHBORS, CFG.batch_size, device)
    indices = entities_inference(topic_ids, bienc, nn_model, topic2text, device, CFG.batch_size)

    t2gold = get_topic_id_gold(corr_df)

    negative_ids = []
    for topic_id, pred_idxs in zip(topic_ids, indices):
        gold = t2gold[topic_id]
        negatives = [i2c[pred_idx] for pred_idx in pred_idxs if i2c[pred_idx] not in gold]
        if not negatives and len(gold) < CFG.NUM_NEIGHBORS:
            raise Exception
        negative_ids.append(" ".join(negatives))
    gen_df = corr_df.copy()
    gen_df["cands"] = negative_ids

    class_ratio = sum(len(x.split()) for x in gen_df["content_ids"]) / sum(len(x.split()) for x in gen_df["cands"])
    print(f"Positive class ratio: {class_ratio}")
    gen_df.to_csv(out_fname, index=False)
    print(f"Saved to: {str(out_fname)}")

    # Log
    run = neptune.init_run(
        project="vmorelli/kolibri",
        source_files=["**/*.py", "*.py"])
    run["bienc_path"] = args.bienc_path
    run["tokenizer_path"] = args.tokenizer_path
    run["num_neigbors"] = CFG.NUM_NEIGHBORS
    run["out"] = args.out
    run["positive_class_ratio"] = class_ratio
    run["part"] = "gen"
    run["bienc_run_id"] = bienc_run_id
