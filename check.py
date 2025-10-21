import torch
print(torch.cuda.is_available())  # True nếu GPU được nhận diện
print(torch.cuda.get_device_name(0))  # Tên GPU
print(torch.__version__)
