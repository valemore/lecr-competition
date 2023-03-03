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
from cross.inference import predict_probs
from cross.model import CrossEncoder
from cross.post import post_process
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
               batch_size: int, device: torch.device) -> pd.DataFrame:
    model = get_cross(cross_dir, device)
    dset = CrossInferenceDataset(cand_df["topic_id"], cand_df["cands"], topic2text, content2text, CFG.CROSS_NUM_TOKENS)
    probs = predict_probs(model, dset, batch_size, device)
    probs = post_process(probs, dset.topic_ids, dset.content_ids)
    preds = (probs >= classifier_thresh).astype(int)
    df = pd.DataFrame({"topic_id": dset.topic_ids, "content_id": dset.content_ids, "pred": preds})
    return df


def to_submission_df_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["pred"] == 1, :]
    df = df.groupby("topic_id").agg(lambda x: " ".join(x)).reset_index(drop=True)
    df = df.rename(columns={"content_id": "content_ids"})
    return df


def main(classifier_thresh: float,
         data_dir: FName, tokenizer_dir: FName, bienc_dir: FName, cross_dir: FName,
         filter_lang: bool,
         bienc_batch_size: int, cross_batch_size: int):
    data_dir = Path(data_dir)
    device = torch.device("cuda")
    init_tokenizer(tokenizer_dir)

    topics_df, content_df, input_df = get_dfs(data_dir, "submit")
    topic_ids = sorted(list(set(input_df["topic_id"])))
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

    df = cross_main(classifier_thresh, cand_df, topic2text, content2text, cross_dir, cross_batch_size, device)
    df = to_submission_df_inplace(df)
    return df
