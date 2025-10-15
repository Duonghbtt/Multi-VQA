# load dataset, EDA, xử lý missing
# Load CSV/JSON, build DataFrame/Dataset object, EDA, xử lý missing
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import ast


class MultiVQADataset(Dataset):
    """
    Dataset cho bài toán Multi-modal VQA (ảnh + hội thoại).
    Hỗ trợ multi-turn conversations và tự load ảnh từ file.
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
        img_dir = os.path.join(split_dir, f"image_{split_name}")

        # Gộp tất cả CSV trong conversation_*
        all_dfs = []
        for file in os.listdir(conv_dir):
            if file.endswith(".csv"):
                df_part = pd.read_csv(os.path.join(conv_dir, file))
                all_dfs.append(df_part)
        if not all_dfs:
            raise ValueError(f"Không tìm thấy CSV nào trong {conv_dir}")

        df = pd.concat(all_dfs, ignore_index=True)

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
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        item = {
            "id": row["id"],
            "image": image,
            "description": row["description"],
            "conversations": row["conversations"],
        }
        return item
