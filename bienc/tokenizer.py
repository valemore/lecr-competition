# Biencoder tokenizer
from typing import Dict, List

from transformers import AutoTokenizer
from config import CFG

tokenizer = None

def init_tokenizer(pretrained_path=None):
    global tokenizer
    if pretrained_path:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(CFG.BIENC_MODEL_NAME, use_fast=True)


def tokenize(text: str, num_tokens: int) -> Dict[str, List[int]]:
    """
    Get input ids and attention mask.
    :param text: text to tokenize
    :param num_tokens: truncate and pad to this many tokens
    :return: dict with input ids and attention mask
    """
    enc = tokenizer(text,
                    max_length=num_tokens,
                    truncation="only_first",
                    padding="max_length",
                    add_special_tokens=True,
                    return_overflowing_tokens=False,
                    return_offsets_mapping=False)

    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
