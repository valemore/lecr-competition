import gc
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import torch
from transformers import AutoModel

from bienc.inference import embed_and_nn, entities_inference, predict_entities, get_cand_df, filter_languages
from bienc.model import BiencoderModule
from bienc.tokenizer import init_tokenizer
from config import CFG
from cross.dset import CrossInferenceDataset
from cross.inference import predict
from cross.model import CrossEncoder
from data.content import get_content2text
from data.topics import get_topic2text
from typehints import FName


def get_test_topic_ids(fname: FName) -> List[str]:
    df = pd.read_csv(fname)
    return sorted(list(set(df["topic_id"])))


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


def get_submission_df(t2preds: Dict[str, Set[str]]) -> pd.DataFrame:
    def to_str(content_ids):
        return " ".join(sorted(list(content_ids)))
    topic_id_col = sorted(t2preds.keys())
    content_ids_col = [to_str(t2preds[topic_id]) for topic_id in topic_id_col]
    df = pd.DataFrame({"topic_id": topic_id_col, "content_ids": content_ids_col})
    return df


def standalone_bienc_main(thresh: float, data_dir: FName, tokenizer_dir: FName, bienc_dir: FName, batch_size: int):
    data_dir = Path(data_dir)
    device = torch.device("cuda")
    init_tokenizer(tokenizer_dir)

    content_df, topics_df, topic_ids, content_ids, c2i, topic2text, content2text = get_data(data_dir)

    distances, indices = bienc_main(topic_ids, content_ids, topic2text, content2text,
                                    bienc_dir, batch_size, device)
    t2preds = predict_entities(topic_ids, distances, indices, thresh, c2i)
    submission_df = get_submission_df(t2preds)
    return submission_df


def bienc_main(topic_ids: List[str], content_ids: List[str],
               topic2text: Dict[str, str], content2text: Dict[str, str], c2i: Dict[str, int],
               filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
               bienc_dir: FName, batch_size: int, device: torch.device):
    encoder = get_bienc(bienc_dir, device)
    nn_model = embed_and_nn(encoder, content_ids, content2text, CFG.NUM_NEIGHBORS, batch_size, device)
    distances, indices = entities_inference(topic_ids, encoder, nn_model, topic2text, device, batch_size)
    # Filter languages
    if filter_lang:
        c2i = c2i.copy()
        c2i["dummy"] = -1
        distances, indices = filter_languages(distances, indices, topic_ids, c2i, t2lang, c2lang)
    return distances, indices


def cross_main(classifier_thresh: float, cand_df: pd.DataFrame, topic2text, content2text, cross_dir: FName,
               batch_size: int, device: torch.device):
    model = get_cross(cross_dir, device)
    dset = CrossInferenceDataset(cand_df["topic_id"], cand_df["cands"], topic2text, content2text, CFG.CROSS_NUM_TOKENS)
    all_preds = predict(model, dset, classifier_thresh, batch_size, device)
    return all_preds


def get_data(data_dir: FName):
    content_df = pd.read_csv(data_dir / "content.csv")
    topics_df = pd.read_csv(data_dir / "topics.csv")
    topic_ids = get_test_topic_ids(data_dir / "sample_submission.csv")
    content_ids = sorted(list(set(content_df["id"])))
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    return content_df, topics_df, topic_ids, content_ids, c2i, topic2text, content2text


def main(classifier_thresh: float,
         data_dir: FName, tokenizer_dir: FName, bienc_dir: FName, cross_dir: FName,
         filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str],
         bienc_batch_size: int, cross_batch_size: int):
    data_dir = Path(data_dir)
    device = torch.device("cuda")
    init_tokenizer(tokenizer_dir)

    content_df, topics_df, topic_ids, content_ids, c2i, topic2text, content2text = get_data(data_dir)

    distances, indices = bienc_main(topic_ids, content_ids,
                                    topic2text, content2text, c2i,
                                    filter_lang, t2lang, c2lang,
                                    bienc_dir, bienc_batch_size, device)
    cand_df = get_cand_df(topic_ids, distances, indices, c2i)
    del distances, indices
    gc.collect()
    all_preds = cross_main(classifier_thresh, cand_df, topic2text, content2text, cross_dir, cross_batch_size, device)

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
