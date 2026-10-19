"""Tells you whether PyTorch can use your NVIDIA GPU. Run:  python check_gpu.py"""
import torch

print("PyTorch version :", torch.__version__)
print("Built with CUDA :", torch.version.cuda)          # None means a CPU-only PyTorch
print("GPU available   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
else:
    print("-> Training will run on the CPU. See docs/VSCODE_GUIDE.md ('Using the GPU').")
