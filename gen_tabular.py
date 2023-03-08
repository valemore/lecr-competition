from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch

from bienc.tokenizer import init_tokenizer
from config import CFG
from data.content import get_content2text
from data.topics import get_topic2text
from submit import bienc_main
from utils import get_dfs, get_content_ids_c2i, get_t2lang_c2lang


def get_tabular_df(topic_ids: List[str], indices, distances, c2i: Dict[str, int], max_num_cands: int = None) -> pd.DataFrame:
    i2c = {content_idx: content_id for content_id, content_idx in c2i.items()}
    all_topic_ids = []
    all_cand_ids = []
    all_ranks = []
    all_dists = []
    for topic_id, pred_idxs, dists in zip(topic_ids, indices, distances):
        if max_num_cands:
            pred_idxs = pred_idxs[:max_num_cands]
        matching_lang = [(i2c[pred_idx], dist) for pred_idx, dist in zip(pred_idxs, dists) if pred_idx != -1]
        assert matching_lang
        cands, dists = zip(*matching_lang)
        all_topic_ids.extend([topic_id] * len(cands))
        all_cand_ids.extend(cands)
        all_ranks.extend(list(range(len(cands))))
        all_dists.extend(dists)
    return pd.DataFrame({"topic_id": all_topic_ids, "cand_id": all_cand_ids, "rank": all_ranks, "dist": all_dists})


if __name__ == "__main__":
    data_dir = Path("../data")
    out_dirs = {
        0: Path("../out/roberta-large-cos10_0307-003338_KLB-772/"),
        1: Path("../out/roberta-large-cos10_0307-003338_KLB-773/"),
        2: Path("../out/roberta-large-cos10_0307-003338_KLB-774/"),
        3: Path("../out/roberta-large-cos10_0307-003338_KLB-775/"),
        4: Path("../out/roberta-large-cos10_0307-003338_KLB-794/")
    }
    cross_df_dir = Path("../cross/roberta-large-cos10_0307-003338/")
    tab_dir = Path("../tab/roberta-large-cos10_0307-003338/")
    num_folds = 5

    device = torch.device("cuda")

    topics_df, content_df, input_df = get_dfs(data_dir, "bienc")
    content_ids, c2i = get_content_ids_c2i(content_df)
    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    t2lang, c2lang = get_t2lang_c2lang(input_df, content_df)


    for fold_idx in range(num_folds):
        out_dir = out_dirs[fold_idx]
        bienc_tokenizer_dir = out_dir / "tokenizer"
        bienc_dir = out_dir / "bienc"
        init_tokenizer(bienc_tokenizer_dir)
        fold_df = pd.read_csv(cross_df_dir / f"fold-{fold_idx}.csv")

        topic_ids = sorted(fold_df["topic_id"])

        indices, distances = bienc_main(topic_ids, content_ids,
                                        topic2text, content2text, c2i,
                                        True, t2lang, c2lang,
                                        bienc_dir, 256, device)

        tab_df = get_tabular_df(topic_ids, indices, distances, c2i, CFG.MAX_NUM_CANDS)
        tab_dir.mkdir(exist_ok=True, parents=True)
        tab_df.to_csv(tab_dir / f"fold-{fold_idx}.csv", index=False)

    tab_df = pd.DataFrame()
    for fold_idx in range(num_folds):
        tab_df = pd.concat([tab_df, pd.read_csv(tab_dir / f"fold-{fold_idx}.csv", keep_default_na=False)]).reset_index(drop=True)
    tab_df = tab_df.sort_values(["topic_id", "rank"]).reset_index(drop=True)
    tab_df.to_csv(tab_dir / "all_folds.csv", index=False)
    print(f'Wrote tabular df to {tab_dir / "roberta-large-cos10_0307-003338" / "all_folds.csv"}')
