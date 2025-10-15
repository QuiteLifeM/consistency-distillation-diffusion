"""
Test script to verify data loading and model setup
"""
import os
import torch
from train import LatentPromptDataset

print("="*60)
print("Testing Data Loading")
print("="*60)

# Paths
latents_dir = r"C:\newTry2\train\datadir\latents"
prompts_dir = r"C:\newTry2\train\datadir\prompts"

# Test dataset
print("\n1. Creating dataset...")
dataset = LatentPromptDataset(latents_dir, prompts_dir)

print(f"\n2. Loading first sample...")
latent, prompt = dataset[0]
print(f"   Latent shape: {latent.shape}")
print(f"   Latent dtype: {latent.dtype}")
print(f"   Prompt: '{prompt}'")

print(f"\n3. Testing multiple samples...")
for i in range(3):
    latent, prompt = dataset[i]
    print(f"   Sample {i}: latent shape={latent.shape}, prompt='{prompt[:50]}...'")

print("\n4. Testing DataLoader...")
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
batch_latents, batch_prompts = next(iter(dataloader))
print(f"   Batch latents shape: {batch_latents.shape}")
print(f"   Batch size: {len(batch_prompts)}")
print(f"   First prompt: '{batch_prompts[0]}'")

print("\n✅ All tests passed!")
print("="*60)




