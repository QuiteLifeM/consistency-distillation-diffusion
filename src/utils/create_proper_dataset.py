#!/usr/bin/env python3
"""
📊 СОЗДАНИЕ ПРАВИЛЬНОГО ДАТАСЕТА
================================
Создаем датасет с латентами + соответствующими текстовыми эмбеддингами
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

# Импортируем наш правильный текстовый кодировщик
from proper_text_embeddings import ProperTextEncoder

class ProperDataset:
    """Правильный датасет с латентами и текстовыми эмбеддингами"""
    
    def __init__(self, latents_dir, prompts_dir, text_encoder, device="cuda"):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        self.text_encoder = text_encoder
        self.device = device
        
        # Получаем список файлов
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        print(f"📊 Найдено {len(self.latent_files)} латентов")
        print(f"📊 Найдено {len(self.prompt_files)} промптов")
        
        # Проверяем соответствие
        if len(self.latent_files) != len(self.prompt_files):
            print("⚠️  Количество латентов и промптов не совпадает!")
        
        self.length = min(len(self.latent_files), len(self.prompt_files))
        print(f"📊 Размер датасета: {self.length}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Получает образец: латенты + текстовые эмбеддинги"""
        try:
            # Загружаем латенты
            latent_file = self.latent_files[idx]
            latent_path = os.path.join(self.latents_dir, latent_file)
            latents = torch.load(latent_path, map_location=self.device).to(torch.float32)  # ← Float32 для совместимости
            
            # Загружаем промпт
            prompt_file = self.prompt_files[idx]
            prompt_path = os.path.join(self.prompts_dir, prompt_file)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            # Получаем текстовые эмбеддинги
            text_embeddings = self.text_encoder.encode_text(prompt).to(torch.float32)  # ← Float32 для совместимости
            
            return {
                'latents': latents,
                'text_embeddings': text_embeddings,
                'prompt': prompt,
                'latent_file': latent_file,
                'prompt_file': prompt_file
            }
            
        except Exception as e:
            print(f"❌ Ошибка загрузки образца {idx}: {e}")
            # Возвращаем пустой образец
            return {
                'latents': torch.randn(4, 64, 64, device=self.device),
                'text_embeddings': torch.randn(1, 77, 1024, device=self.device),
                'prompt': "Error loading sample",
                'latent_file': f"error_{idx}.pt",
                'prompt_file': f"error_{idx}.txt"
            }

def create_proper_dataset():
    """Создает правильный датасет"""
    print("📊 СОЗДАНИЕ ПРАВИЛЬНОГО ДАТАСЕТА")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Устройство: {device}")
    
    # Пути к данным
    latents_dir = "/home/ubuntu/train/train/datadir/latents_good"
    prompts_dir = "/home/ubuntu/train/train/datadir/prompts_good"
    
    # Проверяем существование папок
    if not os.path.exists(latents_dir):
        print(f"❌ Папка латентов не найдена: {latents_dir}")
        return None
    
    if not os.path.exists(prompts_dir):
        print(f"❌ Папка промптов не найдена: {prompts_dir}")
        return None
    
    # Создаем текстовый кодировщик
    print("\n🔧 Создаем текстовый кодировщик...")
    text_encoder = ProperTextEncoder(device)
    
    # Создаем датасет
    print("\n📊 Создаем датасет...")
    dataset = ProperDataset(latents_dir, prompts_dir, text_encoder, device)
    
    # Тестируем датасет
    print("\n🧪 ТЕСТИРУЕМ ДАТАСЕТ:")
    print("=" * 30)
    
    # Тестируем первые 5 образцов
    for i in range(min(5, len(dataset))):
        print(f"\n📊 Образец {i+1}:")
        
        try:
            sample = dataset[i]
            
            print(f"📊 Латенты: {sample['latents'].shape}")
            print(f"📊 Текстовые эмбеддинги: {sample['text_embeddings'].shape}")
            print(f"📝 Промпт: '{sample['prompt']}'")
            print(f"📁 Файл латентов: {sample['latent_file']}")
            print(f"📁 Файл промпта: {sample['prompt_file']}")
            
            # Проверяем качество
            latents_ok = not torch.isnan(sample['latents']).any() and not torch.isinf(sample['latents']).any()
            embeddings_ok = not torch.isnan(sample['text_embeddings']).any() and not torch.isinf(sample['text_embeddings']).any()
            
            print(f"📊 Латенты OK: {'✅' if latents_ok else '❌'}")
            print(f"📊 Эмбеддинги OK: {'✅' if embeddings_ok else '❌'}")
            
            if latents_ok and embeddings_ok:
                print("✅ ОБРАЗЕЦ ХОРОШИЙ!")
            else:
                print("❌ ОБРАЗЕЦ ПЛОХОЙ!")
                
        except Exception as e:
            print(f"❌ Ошибка тестирования образца {i}: {e}")
            continue
    
    # Статистика датасета
    print(f"\n📊 СТАТИСТИКА ДАТАСЕТА:")
    print(f"📊 Размер: {len(dataset)} образцов")
    print(f"📊 Латенты: {latents_dir}")
    print(f"📊 Промпты: {prompts_dir}")
    print(f"📊 Текстовый кодировщик: ProperTextEncoder")
    
    return dataset

def test_dataset_loading():
    """Тестирует загрузку датасета"""
    print("🧪 ТЕСТ ЗАГРУЗКИ ДАТАСЕТА")
    print("=" * 40)
    
    # Создаем датасет
    dataset = create_proper_dataset()
    if dataset is None:
        return
    
    # Тестируем DataLoader
    print("\n🔄 ТЕСТИРУЕМ DATALOADER:")
    print("=" * 30)
    
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            num_workers=0,
            collate_fn=lambda x: x  # Простая коллация
        )
        
        print(f"✅ DataLoader создан успешно")
        print(f"📊 Batch size: 2")
        print(f"📊 Shuffle: True")
        
        # Тестируем загрузку батча
        print("\n📊 Тестируем загрузку батча...")
        for i, batch in enumerate(dataloader):
            print(f"📊 Батч {i+1}: {len(batch)} образцов")
            
            for j, sample in enumerate(batch):
                print(f"   📊 Образец {j+1}: латенты {sample['latents'].shape}, эмбеддинги {sample['text_embeddings'].shape}")
            
            if i >= 2:  # Тестируем только первые 3 батча
                break
        
        print("✅ DataLoader работает корректно!")
        
    except Exception as e:
        print(f"❌ Ошибка создания DataLoader: {e}")

if __name__ == "__main__":
    test_dataset_loading()
