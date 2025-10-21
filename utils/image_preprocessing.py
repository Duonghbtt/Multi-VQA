# -*- coding: utf-8 -*-
# image_preprocessing.py
# Transform ảnh: resize, normalize, convert tensor, optional block splitting (train/test)

from torchvision import transforms
from PIL import Image
import torch

# ==========================
# 1. Transform cho train/test
# ==========================
def get_transform(train=True, image_size=224,
                  mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)):
    """
    Trả về transform pipeline:
    - Nếu train=True: có augmentations (RandomCrop, Flip, ColorJitter)
    - Nếu train=False: chỉ resize + normalize
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),       # resize lớn hơn 1 chút
            transforms.RandomCrop(image_size),               # crop ngẫu nhiên
            transforms.RandomHorizontalFlip(),          # lật ngang
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.1),                # thay đổi nhẹ màu sắc
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform


# ==========================
# 2. Load ảnh + apply transform
# ==========================
def preprocess_image(image_path, train=True, transform=None):
    """
    image_path: đường dẫn ảnh
    train: True -> dùng augmentations
    transform: pipeline có sẵn (nếu None -> tự tạo theo train)
    Trả về: Tensor shape [C, H, W]
    """
    if transform is None:
        transform = get_transform(train=train)
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor


# ==========================
# 3. Test nhanh
# ==========================
if __name__ == "__main__":
    train_tf = get_transform(train=True, image_size=224)
    test_tf = get_transform(train=False, image_size=224)

    print("▶ Train transform:", train_tf)
    print("▶ Test transform:", test_tf)

    img_tensor_train = preprocess_image("example.jpg", train=True, transform=train_tf)
    img_tensor_test = preprocess_image("example.jpg", train=False, transform=test_tf)

    print("Train tensor shape:", img_tensor_train.shape)
    print("Test tensor shape:", img_tensor_test.shape)
