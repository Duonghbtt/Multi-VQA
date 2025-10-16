# resize, normalize, convert tensor
# Transform ảnh (resize, normalize, convert tensor, optional block splitting)
# image_preprocessing.py
# -*- coding: utf-8 -*-
# Transform ảnh: resize, normalize, convert tensor, optional block splitting

from torchvision import transforms
from PIL import Image
import torch

# ==========================
# 1. Transform cơ bản cho CNN/ViT
# ==========================
def get_transform(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Trả về một transform pipeline chuẩn:
    - Resize ảnh về image_size x image_size
    - Convert ảnh thành tensor
    - Normalize
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

# ==========================
# 2. Load ảnh + apply transform
# ==========================
def preprocess_image(image_path, transform=None):
    """
    image_path: đường dẫn ảnh
    transform: transform pipeline (mặc định dùng get_transform)
    Trả về: Tensor shape [C, H, W]
    """
    if transform is None:
        transform = get_transform()
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor

# ==========================
# 4. Test nhanh
# ==========================
if __name__ == "__main__":
    transform = get_transform(image_size=224)
    img_tensor = preprocess_image("example.jpg", transform)
    print("Image tensor shape:", img_tensor.shape)
