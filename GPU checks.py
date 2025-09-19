import torch

print("GPU available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())


print(torch.__version__)
print(torch.version.cuda)   # CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # cuDNN version (if any)
