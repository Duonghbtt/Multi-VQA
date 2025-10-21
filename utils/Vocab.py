from utils.text_preprocessing import preprocess_text
from tqdm import tqdm
import torch

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>":0, "<UNK>":1, "<SOS>":2, "<EOS>":3}
        self.idx2word = {0:"<PAD>", 1:"<UNK>", 2:"<SOS>", 3:"<EOS>"}
        self.idx = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, sentences):
        tokens = set()
        for sent in tqdm(sentences, desc="Tokenizing"):
            try:
                tokens.update(preprocess_text(sent).split())
            except Exception as e:
                print(f"⚠️ Lỗi tokenize câu: {sent[:50]}... → {e}")
        for word in tqdm(tokens, desc="Building Vocab"):
            self.add_word(word)

    # ==================== BỔ SUNG ====================
    def text_to_indices(self, text):
        """Chuyển câu thành list các chỉ số token"""
        tokens = preprocess_text(text).split()
        indices = [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]
        # Thêm token bắt đầu và kết thúc
        indices = [self.word2idx["<SOS>"]] + indices + [self.word2idx["<EOS>"]]
        return indices

    def text_to_tensor(self, text):
        """Chuyển text → tensor"""
        indices = self.text_to_indices(text)
        return torch.tensor(indices, dtype=torch.long)

    def indices_to_text(self, indices):
        """Giải mã ngược lại từ chỉ số → text"""
        words = [self.idx2word.get(int(i), "<UNK>") for i in indices]
        # Bỏ <SOS> và <EOS> khi in ra
        words = [w for w in words if w not in ["<SOS>", "<EOS>", "<PAD>"]]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)
