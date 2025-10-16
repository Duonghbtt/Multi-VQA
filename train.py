# training loop, loss, optimizer, scheduler
# Training loop, loss function, optimizer, scheduler, checkpointing
# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.multimodal_model import VQAModel
from utils.dataset_utils import MultiVQADataset
from inference import Vocabulary

def train(model, dataloader, vocab_size, device, epochs=5, lr=1e-3, save_path="checkpoint.pt"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad token

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, questions, answers in dataloader:
            imgs, questions, answers = imgs.to(device), questions.to(device), answers.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, questions, answer=answers)
            # outputs: [B, max_len, vocab_size], answers: [B, max_len]
            B, T, V = outputs.shape
            loss = criterion(outputs.view(B*T, V), answers.view(B*T))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), save_path)
        print("Checkpoint saved:", save_path)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    train_dataset = MultiVQADataset(r"D:\VQA\data\train", "train", transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 2. Build vocab size
    vocab = Vocabulary()
    all_texts = []
    for item in train_dataset:
        all_texts.append(item["description"])
        for conv in item["conversations"]:
            all_texts.append(conv["content"])
    vocab.build_vocab(all_texts)
    vocab_size = len(vocab.word2idx)
    # 3. Init model
    model = VQAModel(vocab_size=vocab_size, num_classes=vocab_size)

    # 4. Train
    train(model, train_loader, vocab_size, device, epochs=10, lr=1e-3, save_path="checkpoint.pt")
