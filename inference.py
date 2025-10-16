#Demo multi-turn inference, update history, trả lời câu hỏi từ user
# -*- coding: utf-8 -*-
import torch
from models.multimodal_model import VQAModel
from utils import image_preprocessing, text_preprocessing
from utils.dataset_utils import MultiVQADataset
# ==========================
# 1. Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 2. Vocabulary
# ==========================
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
        for sent in sentences:
            for word in text_preprocessing(sent).split():
                self.add_word(word)

vocab = Vocabulary()
# Nếu đã có dataset, bạn có thể gọi vocab.build_vocab(list_of_questions + list_of_answers)
all_texts = []
train_dataset = MultiVQADataset(r"D:\VQA\data\train", "train", transform=None)
for item in train_dataset:
    all_texts.append(item["description"])
    for conv in item["conversations"]:
        all_texts.append(conv["content"])
vocab.build_vocab(all_texts)
# ==========================
# 3. Load Model
# ==========================
vocab_size = len(vocab.word2idx)
model = VQAModel(vocab_size=vocab_size, num_classes=vocab_size)
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.to(device)
model.eval()

# ==========================
# 4. Multi-turn history
# ==========================
history = []

# ==========================
# 5. Inference function
# ==========================
def ask_question(image_path, question, max_len=20):
    global history

    # 5.1 Image preprocessing
    img_tensor = image_preprocessing(image_path).unsqueeze(0).to(device)

    # 5.2 Text preprocessing
    tokens = [vocab.word2idx.get(w, 1) for w in text_preprocessing(question).split()]
    tokens = [vocab.word2idx["<SOS>"]] + tokens
    question_tensor = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]

    # 5.3 Generate answer (token-by-token)
    pred_tokens = model.generate(img_tensor, question_tensor, max_len=max_len)

    # 5.4 Convert indices -> words
    answer_words = []
    for idx in pred_tokens:
        word = vocab.idx2word.get(idx.item(), "<UNK>")
        if word == "<EOS>":
            break
        answer_words.append(word)
    answer = " ".join(answer_words)

    # 5.5 Update history
    history.append((question, answer))
    return answer

# ==========================
# 6. CLI loop
# ==========================
if __name__ == "__main__":
    print("VQA Bot - gõ 'exit' hoặc 'quit' để thoát")
    while True:
        q = input("Bạn hỏi: ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = ask_question("example.jpg", q)
        print("Bot trả lời:", ans)
        print("History:", history)
