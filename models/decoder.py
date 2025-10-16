#Decoder sinh câu trả lời token-by-token
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerDecoder(nn.Module):
    """
    Decoder token-by-token cho VQA.
    Input:
      - fused_feature: tensor [B, fusion_dim]
      - target_seq (optional, cho teacher forcing) [B, seq_len]
    Output:
      - logits [B, seq_len, vocab_size]
    """

    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Input fusion vector → init hidden state
        self.fc_init = nn.Linear(input_dim, hidden_dim)

        # Embedding cho decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, fused_feature, target_seq=None, teacher_forcing_ratio=0.5, max_len=20):
        """
        fused_feature: [B, input_dim]
        target_seq: [B, seq_len] hoặc None khi inference
        """
        B = fused_feature.size(0)
        device = fused_feature.device

        # khởi tạo hidden state từ fused feature
        h0 = torch.tanh(self.fc_init(fused_feature)).unsqueeze(0)  # [1, B, hidden_dim]
        c0 = torch.zeros_like(h0)  # [1, B, hidden_dim]

        # nếu có target_seq → teacher forcing
        if target_seq is not None:
            seq_len = target_seq.size(1)
            inputs = self.embedding(target_seq)  # [B, seq_len, hidden_dim]
            outputs, _ = self.lstm(inputs, (h0, c0))  # [B, seq_len, hidden_dim]
            logits = self.fc_out(outputs)  # [B, seq_len, vocab_size]
            return logits

        # inference: sinh token-by-token
        inputs = torch.full((B, 1), 1, dtype=torch.long, device=device)  # assume <SOS>=1
        outputs_list = []

        hidden = (h0, c0)
        for t in range(max_len):
            emb = self.embedding(inputs)  # [B, 1, hidden_dim]
            out, hidden = self.lstm(emb, hidden)  # out: [B, 1, hidden_dim]
            logits = self.fc_out(out)  # [B, 1, vocab_size]
            outputs_list.append(logits)
            # chọn token max
            inputs = logits.argmax(-1)

        outputs = torch.cat(outputs_list, dim=1)  # [B, max_len, vocab_size]
        return outputs
