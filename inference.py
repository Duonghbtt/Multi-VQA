# -*- coding: utf-8 -*-
# inference.py
# Demo multi-turn inference, update history, trả lời câu hỏi từ user

import torch
from models.multimodal_model import VQAModel
from utils.image_preprocessing import preprocess_image
from utils.text_preprocessing import normalize_text
from utils.dataset_utils import MultiVQADataset
from utils.Vocab import  Vocabulary
# ==========================
# 1. Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 3. Load vocab & dataset
# ==========================
vocab = Vocabulary()
train_dataset = MultiVQADataset(r"D:\VQA\data\test", "test", transform=None)
all_texts = []
for item in train_dataset:
    all_texts.append(item["description"])
    for conv in item["conversations"]:
        all_texts.append(conv["q"])
        all_texts.append(conv["a"])
vocab.build_vocab(all_texts)
vocab_size = len(vocab.word2idx)

# ==========================
# 4. Load Model
# ==========================
vocab_size = len(vocab.word2idx)
model = VQAModel(vocab_size=vocab_size, num_classes=vocab_size)
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.to(device)
model.eval()

# ==========================
# 5. Multi-turn history
# ==========================
history = []

# ==========================
# 6. Inference function
# ==========================
def ask_question(image_path, question, max_len=20):
    global history

    # 6.1 Ảnh
    img_tensor = preprocess_image(image_path, train=False).unsqueeze(0).to(device)

    # 6.2 Câu hỏi
    tokens = [vocab.word2idx.get(w, 1) for w in normalize_text(question).split()]
    tokens = [vocab.word2idx["<SOS>"]] + tokens
    question_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # 6.3 Sinh câu trả lời
    pred_tokens = model.generate(img_tensor, question_tensor, max_len=max_len)

    # 6.4 Chuyển chỉ số -> từ
    answer_words = []
    for idx in pred_tokens:
        word = vocab.idx2word.get(idx.item(), "<UNK>")
        if word == "<EOS>":
            break
        answer_words.append(word)
    answer = " ".join(answer_words)

    # 6.5 Cập nhật history
    history.append((question, answer))
    return answer


# ==========================
# 7. CLI loop
# ==========================
if __name__ == "__main__":
    print("🤖 VQA Bot — gõ 'exit' hoặc 'quit' để thoát.")
    while True:
        q = input("Bạn hỏi: ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = ask_question("example.jpg", q)
        print("Bot trả lời:", ans)
        print("Lịch sử hội thoại:", history)
