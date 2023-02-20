# Crossencoder tkenizer
# from typing import Dict, List
#
# from transformers import PreTrainedTokenizer, AutoTokenizer
# from config import CFG
#
# tokenizer = None
#
# def init_tokenizer(pretrained_path=None):
#     global tokenizer
#     if pretrained_path:
#         tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(CFG.CROSSENC_MODEL_NAME, use_fast=True)
#
#
# def tokenize(topic_text: str, content_text, num_tokens: int) -> Dict[str, List[int]]:
#     """
#     Get input ids and attention mask.
#     :param topic_text: text to tokenize
#     :param num_tokens: truncate and pad to this many tokens
#     :return: dict with input ids and attention mask
#     """
#     enc = tokenizer(topic_text, content_text,
#                     max_length=num_tokens,
#                     truncation="only_first",
#                     padding="max_length",
#                     add_special_tokens=True,
#                     return_overflowing_tokens=False,
#                     return_offsets_mapping=False)
#
#     return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
