# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.amp import autocast, GradScaler
from models.multimodal_model import VQAModel
from utils.dataset_utils import MultiVQADataset
from utils.Vocab import Vocabulary
from tqdm import tqdm
import os
from functools import partial
import gc
# ============================================================
# ⚙️ CẤU HÌNH HIỆU NĂNG
# ============================================================
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.benchmark = True


# ============================================================
# 🧩 Dataset Fine-Tune
# ============================================================
class VQAFineTuneDataset(Dataset):
    def __init__(self, base_dataset, vocab, transform=None):
        self.base_dataset = base_dataset
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item["image"]
        conv_list = item["conversations"]

        if self.transform and image is not None:
            image = self.transform(image)

        q_list, a_list = [], []
        for pair in conv_list:
            q = pair.get("q", "").strip()
            a = pair.get("a", "").strip()
            if q and a:
                q_idx = [self.vocab.word2idx.get(w, self.vocab.word2idx["<UNK>"]) for w in q.split()]
                a_idx = [self.vocab.word2idx.get(w, self.vocab.word2idx["<UNK>"]) for w in a.split()]
                q_list.append(torch.tensor(q_idx, dtype=torch.long))
                a_list.append(torch.tensor(a_idx, dtype=torch.long))
        return image, q_list, a_list

    def __len__(self):
        return len(self.base_dataset)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 🧮 Collate Function
# ============================================================
def collate_fn(batch, pad_idx=0):
    imgs, all_qs, all_as = [], [], []
    for img, q_list, a_list in batch:
        # Lặp ảnh để mỗi câu hỏi có ảnh tương ứng
        imgs.extend([img] * len(q_list))
        all_qs.extend(q_list)
        all_as.extend(a_list)

    imgs = torch.stack(imgs)
    qs = torch.nn.utils.rnn.pad_sequence(all_qs, batch_first=True, padding_value=pad_idx)
    ans = torch.nn.utils.rnn.pad_sequence(all_as, batch_first=True, padding_value=pad_idx)
    return imgs, qs, ans


# ============================================================
# 🚀 Train + Validation
# ============================================================
def train_finetune(model_class, dataset, vocab, device, pretrain_path=None,
                   epochs=10, lr=1e-4, batch_size=2, val_ratio=0.1):

    # Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Split train/val
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    print(f"📊 Train samples: {train_size} | Val samples: {val_size}")

    train_data = VQAFineTuneDataset(train_set, vocab, transform=train_transform)
    val_data = VQAFineTuneDataset(val_set, vocab, transform=val_transform)

    pad_idx = vocab.word2idx["<PAD>"]

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate_fn, pad_idx=pad_idx)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate_fn, pad_idx=pad_idx)
    )

    # Model
    model = model_class(vocab_size=len(vocab.word2idx)).to(device)

    # Load pretrain weights
    if pretrain_path and os.path.exists(pretrain_path):
        state_dict = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrain weights from {pretrain_path}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_model_path = "checkpoint_best.pt"

    # Training Loop
    for epoch in range(epochs):
        cleanup()
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for imgs, qs, ans in train_bar:
            imgs, qs, ans = imgs.to(device, non_blocking=True), qs.to(device, non_blocking=True), ans.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                #with autocast(device_type="cuda"):
                outputs = model(imgs, qs, answer=ans)
                B, T, V = outputs.shape
                loss = criterion(outputs.view(B * T, V), ans.view(B * T))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            except torch.cuda.OutOfMemoryError:
                print("⚠️ CUDA OOM — Skipping batch.")
                cleanup()
                continue
        avg_train_loss = total_loss / len(train_loader)
        cleanup()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():#, autocast(device_type="cuda"):
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
            for imgs, qs, ans in val_bar:
                imgs, qs, ans = imgs.to(device), qs.to(device), ans.to(device)
                try:
                    outputs = model(imgs, qs, answer=ans)
                    B, T, V = outputs.shape
                    loss = criterion(outputs.view(B * T, V), ans.view(B * T))
                    val_loss += loss.item()
                except torch.cuda.OutOfMemoryError:
                    print("⚠️ OOM during validation batch — skipped")
                    cleanup()
                    continue
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        cleanup()
        print(f"📘 Epoch {epoch+1}/{epochs} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | LR={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model: {best_model_path} (ValLoss={best_val_loss:.4f})")

    print(f"\n🏆 Finished! Best Val Loss = {best_val_loss:.4f}")
    print(f"💾 Final checkpoint saved at: {best_model_path}")


# ============================================================
# 🏁 Main Entry
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Device: {device}")

    train_dataset = MultiVQADataset(r"D:\VQA\data\train", "train")
    # vocab = Vocabulary()
    # all_texts = df["description"].tolist()
    #
    # all_questions, all_answers = [], []
    # for conv_list in df["conversations"]:
    #     for i in range(len(conv_list)-1):
    #         if conv_list[i]["role"] == "user" and conv_list[i+1]["role"] == "assistant":
    #             q = conv_list[i]["content"].strip()
    #             a = conv_list[i+1]["content"].strip()
    #             if q:
    #                 all_questions.append(q)
    #             if a:
    #                 all_answers.append(a)
    #
    # all_texts.extend(all_questions)
    # all_texts.extend(all_answers)
    #
    # vocab.build_vocab(all_texts)
    # torch.save(vocab, r"D:\VQA\utils\vocab.pt")
    torch.serialization.add_safe_globals([Vocabulary])
    vocab = torch.load(r"D:\VQA\utils\vocab.pt", weights_only=False)
    print(f"✅ Vocabulary size: {len(vocab.word2idx)} tokens")
    print("Sample tokens:", list(vocab.word2idx.keys())[:20])

    pretrain_path = r"D:\VQA\checkpoints\pretrained_best.pt"

    train_finetune(
        model_class=VQAModel,
        dataset=train_dataset,
        vocab=vocab,
        device=device,
        pretrain_path=pretrain_path,
        epochs=10,
        lr=1e-4,
        batch_size=1,
        val_ratio=0.1
    )
