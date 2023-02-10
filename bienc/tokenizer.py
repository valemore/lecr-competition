# Tokenizer

from transformers import PreTrainedTokenizer, AutoTokenizer
from config import BIENC_MODEL_NAME


tokenizer = None

def init_tokenizer(pretrained_path=None):
    global tokenizer
    if pretrained_path:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BIENC_MODEL_NAME, use_fast=True)
