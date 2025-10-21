# load dataset, EDA, xử lý missing
# Load CSV/JSON, build DataFrame/Dataset object, EDA, xử lý missing
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import ast
from torchvision import transforms
from utils.text_preprocessing import preprocess_text
class MultiVQADataset(Dataset):
    """
    Dataset cho bài toán Multi-modal VQA (ảnh + hội thoại).
    Format hội thoại: [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
    """

    def __init__(self, split_dir: str, split_name: str, transform=None):
        """
        Args:
            split_dir (str): thư mục gốc của data/train hoặc data/test.
            split_name (str): 'train' hoặc 'test'.
            transform: torchvision transforms cho ảnh (nếu có).
        """
        self.split_name = split_name
        self.transform = transform

        conv_dir = os.path.join(split_dir, f"conversation_{split_name}")
        img_dir = os.path.join(split_dir, f"images_{split_name}")

        # Gộp tất cả CSV trong conversation_*
        all_dfs = []
        for file in os.listdir(conv_dir):
            if file.endswith(".csv"):
                df_part = pd.read_csv(os.path.join(conv_dir, file))
                all_dfs.append(df_part)
        if not all_dfs:
            raise ValueError(f"Không tìm thấy CSV nào trong {conv_dir}")
        print(f"Số lượng csv trong {split_name} là {len(all_dfs)}")
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Tổng số dòng (mẫu) trong {split_name}: {df.shape[0]}")
        print(f"Số cột trong {split_name}: {df.shape[1]}")

        # Làm sạch và xử lý cột
        df.dropna(subset=["id", "description"], inplace=True)
        df.drop_duplicates(subset="id", inplace=True)

        # Parse conversation (nếu là chuỗi JSON)
        def safe_parse(x):
            if isinstance(x, list):
                return x
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        df["conversations"] = df["conversations"].apply(safe_parse)

        # Gắn đường dẫn ảnh
        df["image_path"] = df["id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))

        # Giữ lại các dòng có ảnh tồn tại
        df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        description = preprocess_text(row["description"])

        conversations = []
        convs = row["conversations"]

        # Lặp qua toàn bộ hội thoại → lấy từng cặp (user → assistant)
        for i in range(len(convs) - 1):
            if convs[i].get("role") == "user" and convs[i + 1].get("role") == "assistant":
                q = convs[i].get("content", "")
                a = convs[i + 1].get("content", "")
                conversations.append({
                    "q": preprocess_text(q),
                    "a": preprocess_text(a)
                })

        item = {
            "id": row["id"],
            "image": image,
            "description": description,
            "conversations": conversations
        }
        return item

if __name__ == "__main__":
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    dataset = MultiVQADataset(
        split_dir=r"D:\VQA\data\train",
        split_name="train",
        transform=train_transform
    )

    print("Tổng mẫu:", len(dataset))
    sample = dataset[0]
    print("🆔 ID:", sample["id"])
    print("❓ Q:", sample["image"])
    print("❓ D:", sample["description"])
    print("💬 A:", sample["conversations"])