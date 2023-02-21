import torch.nn as nn
from transformers import AutoModel

from config import CFG

class CrossEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        if pretrained_path:
            self.encoder =  AutoModel.from_pretrained(pretrained_path)
        else:
            self.encoder = AutoModel.from_pretrained(CFG.CROSS_MODEL_NAME)
        self.linear = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        out = out.last_hidden_state
        # CLS pooling
        out = out[:, 0, :] # TODO: ???
        out = self.linear(out)
        return out
