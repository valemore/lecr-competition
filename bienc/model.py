# Bi-encoder model definition
import torch
import torch.nn as nn
from transformers import AutoModel

from config import CFG


class Biencoder(nn.Module):
    def __init__(self, score_fn, pretrained_path=None):
        super().__init__()
        self.topic_encoder = BiencoderModule(pretrained_path)
        self.content_encoder = self.topic_encoder
        self.score_fn = score_fn

    def forward(self, topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask):
        topic_emb = self.topic_encoder(topic_input_ids, topic_attention_mask)
        content_emb = self.content_encoder(content_input_ids, content_attention_mask)
        return self.score_fn(topic_emb, content_emb)


class BiencoderModule(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        if pretrained_path:
            self.encoder =  AutoModel.from_pretrained(pretrained_path)
        else:
            self.encoder = AutoModel.from_pretrained(CFG.BIENC_MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        out = out.last_hidden_state
        mask_sum = torch.clamp(attention_mask.sum(dim=-1), min=1e-9)
        # Mean pooling
        out = (out * attention_mask.unsqueeze(-1).expand(out.shape)).sum(dim=1) / mask_sum.unsqueeze(-1)
        return out
