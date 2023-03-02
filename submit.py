import gc
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import torch

from bienc.gen_cands import get_cand_df
from bienc.inference import do_nn
from bienc.model import BiencoderModule
from bienc.tokenizer import init_tokenizer
from config import CFG
from cross.dset import CrossInferenceDataset
from cross.inference import predict
from cross.model import CrossEncoder
from data.content import get_content2text
from data.topics import get_topic2text
from typehints import FName
from utils import get_t2lang_c2lang, get_dfs, get_content_ids_c2i


def get_bienc(bienc_dir: FName, device: torch.device) -> BiencoderModule:
    model = BiencoderModule(bienc_dir)
    model.to(device)
    model.eval()
    return model


def get_cross(cross_dir: FName, device: torch.device):
    model = CrossEncoder(dropout=CFG.cross_dropout, save_dir=cross_dir).to(device)
    model.load(cross_dir)
    model.eval()
    return model


def bienc_main(topic_ids: List[str], content_ids: List[str],
               topic2text: Dict[str, str], content2text: Dict[str, str], c2i: Dict[str, int],
               filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
               bienc_dir: FName, batch_size: int, device: torch.device):
    encoder = get_bienc(bienc_dir, device)
    indices = do_nn(encoder, topic_ids, content_ids, topic2text, content2text,
                    filter_lang, t2lang, c2lang, c2i,
                    batch_size, device)
    return indices


def cross_main(classifier_thresh: float, cand_df: pd.DataFrame, topic2text, content2text, cross_dir: FName,
               batch_size: int, device: torch.device):
    model = get_cross(cross_dir, device)
    dset = CrossInferenceDataset(cand_df["topic_id"], cand_df["cands"], topic2text, content2text, CFG.CROSS_NUM_TOKENS)
    all_preds = predict(model, dset, classifier_thresh, batch_size, device)
    return all_preds


def get_submission_df(t2preds: Dict[str, Set[str]]) -> pd.DataFrame:
    def to_str(content_ids):
        return " ".join(sorted(list(content_ids)))
    topic_id_col = sorted(t2preds.keys())
    content_ids_col = [to_str(t2preds[topic_id]) for topic_id in topic_id_col]
    df = pd.DataFrame({"topic_id": topic_id_col, "content_ids": content_ids_col})
    return df


def main(classifier_thresh: float,
         data_dir: FName, tokenizer_dir: FName, bienc_dir: FName, cross_dir: FName,
         filter_lang: bool,
         bienc_batch_size: int, cross_batch_size: int):
    data_dir = Path(data_dir)
    device = torch.device("cuda")
    init_tokenizer(tokenizer_dir)

    topics_df, content_df, input_df = get_dfs(data_dir, "submit")
    topic_ids = sorted(input_df["topic_id"])
    content_ids, c2i = get_content_ids_c2i(content_df)
    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    t2lang, c2lang = get_t2lang_c2lang(input_df, content_df)

    indices = bienc_main(topic_ids, content_ids,
                         topic2text, content2text, c2i,
                         filter_lang, t2lang, c2lang,
                         bienc_dir, bienc_batch_size, device)
    cand_df = get_cand_df(topic_ids, indices, c2i)
    del indices
    gc.collect()
    all_preds = cross_main(classifier_thresh, cand_df, topic2text, content2text, cross_dir, cross_batch_size, device)

    # TODO
    t2preds = {}
    for topic_id in topic_ids:
        t2preds[topic_id] = set()
    i = 0
    for topic_id, topic_cand_ids in zip(cand_df["topic_id"], cand_df["cands"]):
        for cand_id in topic_cand_ids.split():
            if all_preds[i]:
                t2preds[topic_id].add(cand_id)
            i += 1

    submission_df = get_submission_df(t2preds)
    return submission_df
