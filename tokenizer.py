# Tokenizer

from transformers import PreTrainedTokenizer, AutoTokenizer
from config import BIENC_MODEL_NAME


tokenizer = AutoTokenizer.from_pretrained(BIENC_MODEL_NAME, use_fast=True)
