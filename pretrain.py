from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.dataset_utils import MultiVQADataset
from utils.Vocab import Vocabulary
from models.multimodal_model import VQAModel
import os
from torch.amp import autocast, GradScaler

# ======================== CẤU HÌNH HIỆU NĂNG ========================
torch.backends.cuda.matmul.fp32_precision = "tf32"  # TF32 matmul
torch.backends.cudnn.conv.fp32_precision = "tf32"   # TF32 conv
torch.backends.cudnn.benchmark = True               # Auto chọn thuật toán nhanh

# ======================== DATASET TOKENIZED ========================
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, vocab, transform=None):
        self.base_dataset = base_dataset
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = item["image"]
        caption = item["description"]

        if self.transform and img is not None:
            img = self.transform(img)

        tokens = caption.split()
        caption_tensor = torch.tensor(
            [self.vocab.word2idx.get(w, self.vocab.word2idx["<UNK>"]) for w in tokens],
            dtype=torch.long
        )
        return img, caption_tensor

    def __len__(self):
        return len(self.base_dataset)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    caps = torch.nn.utils.rnn.pad_sequence(
        caps, batch_first=True, padding_value=0  # <PAD>=0
    )
    return imgs, caps

# ======================== TRAIN FUNCTION ========================
def train_pretrain(model_class, dataset, vocab, device, epochs=10, lr=1e-3, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # giảm độ phân giải để tăng tốc
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    cap_dataset = CaptionDataset(dataset, vocab=vocab, transform=transform)

    val_ratio = 0.1
    val_size = int(len(cap_dataset) * val_ratio)
    train_size = len(cap_dataset) - val_size
    train_set, val_set = random_split(cap_dataset, [train_size, val_size])

    # Windows: num_workers=0-2
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = model_class(vocab_size=len(vocab.word2idx)).to(device)

    # Freeze image encoder để giảm VRAM và tăng tốc
    if hasattr(model, "img_encoder"):
        for p in model.img_encoder.parameters():
            p.requires_grad = False
        print("🧊 Đã freeze image encoder (chỉ train decoder).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"]).to(device)
    scaler = GradScaler()  # AMP

    best_val_loss = float("inf") # 5.1337
    print("\n🚀 Bắt đầu pretrain captioning...\n")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100)

        for imgs, captions in train_bar:
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                outputs = model(imgs, question=inputs)
                min_T = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_T, :]
                targets = targets[:, :min_T]
                loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", ncols=100)
        with torch.no_grad(), autocast(device_type="cuda"):
            for imgs, captions in val_bar:
                imgs = imgs.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                inputs = captions[:, :-1]
                targets = captions[:, 1:]

                outputs = model(imgs, question=inputs)
                min_T = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_T, :]
                targets = targets[:, :min_T]
                loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
                total_val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\n📘 Epoch {epoch+1}/{epochs} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        # SAVE BEST + LAST
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "pretrained_best.pt"))
            print(f"\n🏁 Best Val Loss = {best_val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, "pretrained_last.pt"))

    print(f"\n🏁 Training done! Best Val Loss = {best_val_loss:.4f}")


# ======================== MAIN ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Device: {device}")

    dataset = MultiVQADataset(r"D:\VQA\data\train", "train")

    torch.serialization.add_safe_globals([Vocabulary])
    vocab = torch.load(r"D:\VQA\utils\vocab.pt", weights_only=False)

    train_pretrain(VQAModel, dataset, vocab, device, epochs=10, lr=1e-3)
