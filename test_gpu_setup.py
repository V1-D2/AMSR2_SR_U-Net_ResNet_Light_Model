#!/usr/bin/env python3
"""
Quick test script to verify GPU setup and data loading
"""

import torch
import numpy as np
import os
import glob

print("=" * 60)
print("AMSR2 GPU SETUP TEST")
print("=" * 60)

# 1. Check PyTorch and GPU
print("\n1. PyTorch & GPU Check:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Test GPU operations
    try:
        x = torch.randn(4, 1, 2000, 420).cuda()
        print(f"   ✅ GPU tensor creation successful")
        print(f"   Test tensor shape: {x.shape}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ GPU tensor creation failed: {e}")
else:
    print("   ⚠️  No GPU available, will use CPU")

# 2. Check data directory
print("\n2. Data Directory Check:")
data_dir = "/home/vdidur/temperature_sr_project/data"

if os.path.exists(data_dir):
    print(f"   ✅ Data directory exists: {data_dir}")

    # Count NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"   Found {len(npz_files)} NPZ files")

    if npz_files:
        # Check first file
        first_file = npz_files[0]
        print(f"   First file: {os.path.basename(first_file)}")

        try:
            with np.load(first_file, allow_pickle=True) as data:
                if 'swath_array' in data:
                    swath_array = data['swath_array']
                    print(f"   ✅ File structure valid")
                    print(f"   Number of swaths: {len(swath_array)}")
                else:
                    print(f"   ❌ Invalid file structure (no swath_array)")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
else:
    print(f"   ❌ Data directory not found: {data_dir}")

# 3. Test mixed precision
print("\n3. Mixed Precision Test:")
if torch.cuda.is_available():
    try:
        with torch.cuda.amp.autocast():
            x = torch.randn(2, 1, 512, 512).cuda()
            y = x * 2.0
        print("   ✅ Mixed precision (AMP) working")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Mixed precision failed: {e}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)