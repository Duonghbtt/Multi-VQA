# Xây dựng LSTM encoder cho text
import torch
import torchvision.models as models
import torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)


    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embed(x)
        _, (h_n, _) = self.lstm(emb)
        return h_n[-1]  # (B, hidden_dim)
