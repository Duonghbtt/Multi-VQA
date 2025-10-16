import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.decoder import AnswerDecoder

class VQAModel(nn.Module):
    def __init__(self, vocab_size, img_feat_dim=2048, txt_hidden_dim=512, decoder_hidden_dim=512, fusion_dim=None):
        super().__init__()
        # 1️⃣ Encoders
        self.img_encoder = ImageEncoder(output_dim=img_feat_dim)
        self.txt_encoder = TextEncoder(vocab_size, hidden_dim=txt_hidden_dim)

        # 2️⃣ Fusion
        self.fusion_dim = fusion_dim or (img_feat_dim + txt_hidden_dim)

        # 3️⃣ Decoder token-by-token
        self.decoder = AnswerDecoder(
            input_dim=self.fusion_dim,
            hidden_dim=decoder_hidden_dim,
            vocab_size=vocab_size
        )

    def forward(self, img, question, answer=None, teacher_forcing_ratio=0.5, max_len=20):
        """
        img: [B, 3, H, W]
        question: [B, q_len]
        answer: [B, a_len] (teacher forcing)
        """
        B = img.size(0)

        # -------- Encode
        img_feat = self.img_encoder(img)      # [B, img_feat_dim]
        txt_feat = self.txt_encoder(question) # [B, txt_hidden_dim]

        # -------- Fusion
        fused_feat = torch.cat([img_feat, txt_feat], dim=1)  # [B, fusion_dim]

        # -------- Decode
        logits = self.decoder(
            fused_feat,
            target_seq=answer,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_len=max_len
        )
        return logits

    def generate(self, img, question, max_len=20):
        """Sinh token-by-token (inference)"""
        self.eval()
        with torch.no_grad():
            # Encode
            img_feat = self.img_encoder(img)
            txt_feat = self.txt_encoder(question)
            fused_feat = torch.cat([img_feat, txt_feat], dim=1)

            # Decode
            pred_logits = self.decoder(fused_feat, target_seq=None, max_len=max_len)
            pred_tokens = pred_logits.argmax(-1)
        return pred_tokens
