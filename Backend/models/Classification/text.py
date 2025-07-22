import torch
print(torch.cuda.is_available())  # Should return: True
print(torch.cuda.get_device_name(0))  # Should return something like: NVIDIA GeForce RTX 4050
