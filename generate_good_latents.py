import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def generate_good_latents(latents_dir, num_samples=100):
    """Генерируем хорошие латенты без NaN"""
    print(f"Генерируем {num_samples} хороших латентов...")
    
    # Создаем директорию если не существует
    os.makedirs(latents_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Генерируем случайный латент без NaN
        latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        
        # Убеждаемся, что нет NaN
        assert not torch.isnan(latent).any(), f"NaN в латенте {i}"
        
        # Сохраняем
        filename = f"image_{i+1:05d}.pt"
        filepath = os.path.join(latents_dir, filename)
        torch.save(latent, filepath)
        
        if (i + 1) % 10 == 0:
            print(f"Создано {i+1}/{num_samples} латентов")
    
    print(f"Создано {num_samples} хороших латентов в {latents_dir}")

def generate_good_prompts(prompts_dir, num_samples=100):
    """Генерируем промпты для латентов"""
    print(f"Генерируем {num_samples} промптов...")
    
    # Создаем директорию если не существует
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Список простых промптов
    prompts = [
        "a beautiful landscape",
        "a cute cat",
        "a red car",
        "a blue sky",
        "a green tree",
        "a yellow flower",
        "a white house",
        "a black dog",
        "a brown bear",
        "a purple butterfly"
    ]
    
    for i in range(num_samples):
        # Выбираем случайный промпт
        prompt = prompts[i % len(prompts)]
        
        # Сохраняем
        filename = f"image_{i+1:05d}.txt"
        filepath = os.path.join(prompts_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        if (i + 1) % 10 == 0:
            print(f"Создано {i+1}/{num_samples} промптов")
    
    print(f"Создано {num_samples} промптов в {prompts_dir}")

if __name__ == "__main__":
    # Пути к данным
    latents_dir = r"C:\newTry2\train\datadir\latents_good"
    prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    # Генерируем хорошие данные
    generate_good_latents(latents_dir, num_samples=100)
    generate_good_prompts(prompts_dir, num_samples=100)
    
    print("\nГотово! Теперь можно использовать:")
    print(f"   latents_dir = r'{latents_dir}'")
    print(f"   prompts_dir = r'{prompts_dir}'")
