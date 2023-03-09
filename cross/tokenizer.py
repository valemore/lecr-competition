# Crossencoder tokenizer
from typing import Dict, List

from transformers import AutoTokenizer
from config import CFG

tokenizer = None
MASK_TOKEN_ID = None

def init_tokenizer(pretrained_path=None):
    global tokenizer, MASK_TOKEN_ID
    if pretrained_path:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
        MASK_TOKEN_ID = tokenizer.mask_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(CFG.CROSS_MODEL_NAME, use_fast=True)
        MASK_TOKEN_ID = tokenizer.mask_token_id


def tokenize_cross(topic_text: str, content_text, num_tokens: int) -> Dict[str, List[int]]:
    """
    Get input ids and attention mask.
    :param topic_text: topic text to tokenize
    :param content_text: content text to tokenize
    :param num_tokens: truncate and pad to this many tokens
    :return: dict with input ids and attention mask
    """
    enc = tokenizer(topic_text, content_text,
                    max_length=num_tokens,
                    truncation="longest_first",
                    padding="max_length",
                    add_special_tokens=True,
                    return_overflowing_tokens=False,
                    return_offsets_mapping=False,
                    return_special_tokens_mask=True,
                    return_tensors="pt")

    return enc.input_ids.reshape(-1), enc.attention_mask.reshape(-1), enc.special_tokens_mask.reshape(-1)
