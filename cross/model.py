from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from config import CFG

class CrossEncoder(nn.Module):
    def __init__(self, dropout=0.1, save_dir=None):
        super().__init__()
        if save_dir:
            config = AutoConfig.from_pretrained(save_dir)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(CFG.CROSS_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        out = out.last_hidden_state
        # CLS pooling
        # TODO: Other pooling strategies
        out = out[:, 0, :]
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def load(self, save_dir):
        save_dir = Path(save_dir)
        self.load_state_dict(torch.load(save_dir / "state_dict.pt"))

    def save(self, save_dir):
        save_dir = Path(save_dir)
        self.encoder.config.save_pretrained(save_dir)
        torch.save(self.state_dict(), save_dir / "state_dict.pt")
        print(f"Model saved to {save_dir}")
