import torch
import torch.nn as nn
from transformers import AutoModel

from config import BIENC_MODEL_NAME


class Biencoder(nn.Module):
    def __init__(self, score_fn):
        super().__init__()
        self.content_encoder = BiencoderModule()
        self.topic_encoder = self.content_encoder
        self.score_fn = score_fn

    def forward(self, query_input_ids, query_attention_mask, ent_input_ids, ent_attention_mask):
        query_emb = self.content_encoder(query_input_ids, query_attention_mask)
        ent_emb = self.topic_encoder(ent_input_ids, ent_attention_mask)
        return self.score_fn(query_emb, ent_emb)


class BiencoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BIENC_MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        out = out.last_hidden_state
        mask_sum = torch.clamp(attention_mask.sum(dim=-1), min=1e-9)
        # Mean pooling
        out = (out * attention_mask.unsqueeze(-1).expand(out.shape)).sum(dim=1) / mask_sum.unsqueeze(-1)
        return out
