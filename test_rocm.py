#!/usr/bin/env python3

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "No CUDA")
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU detected:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test basic tensor operations on GPU
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("GPU tensor operations working!")
        print(f"Test tensor device: {z.device}")
    except Exception as e:
        print(f"GPU tensor operations failed: {e}")
else:
    print("No GPU detected.")
    print("Possible issues:")
    print("1. ROCm PyTorch not properly installed")
    print("2. Environment variables not set correctly") 
    print("3. GPU not supported by current ROCm version")
    print("4. Using CUDA PyTorch instead of ROCm PyTorch")
