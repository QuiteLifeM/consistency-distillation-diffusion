"""
Простой тест без загрузки моделей - только проверка данных
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader

class LatentPromptDataset(Dataset):
    """Dataset для загрузки предвычисленных латентов и промптов"""
    def __init__(self, latents_dir, prompts_dir):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        
        # Получаем отсортированные списки файлов
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        assert len(self.latent_files) == len(self.prompt_files), \
            f"Количество латентов ({len(self.latent_files)}) != количество промптов ({len(self.prompt_files)})"
        
        print(f"✅ Загружено {len(self.latent_files)} латентов и промптов")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # Загружаем латент
        latent_path = os.path.join(self.latents_dir, self.latent_files[idx])
        latent = torch.load(latent_path)
        
        # Загружаем промпт
        prompt_path = os.path.join(self.prompts_dir, self.prompt_files[idx])
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        return latent, prompt

def custom_collate(batch):
    """Кастомная функция для обработки батчей"""
    latents_list = []
    prompts_list = []
    
    for latent, prompt in batch:
        # Убеждаемся, что латент имеет правильную размерность [C, H, W]
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # Убираем лишнюю batch dimension
        
        latents_list.append(latent)
        prompts_list.append(prompt)
    
    # Стакаем латенты в батч [B, C, H, W]
    latents_batch = torch.stack(latents_list, dim=0)
    
    return latents_batch, prompts_list

if __name__ == "__main__":
    print("="*60)
    print("Простой тест загрузки данных (без моделей)")
    print("="*60)
    
    # Пути к данным
    latents_dir = r"C:\newTry2\train\datadir\latents"
    prompts_dir = r"C:\newTry2\train\datadir\prompts"
    
    # Проверяем, что папки существуют
    if not os.path.exists(latents_dir):
        print(f"❌ Папка латентов не найдена: {latents_dir}")
        exit(1)
    
    if not os.path.exists(prompts_dir):
        print(f"❌ Папка промптов не найдена: {prompts_dir}")
        exit(1)
    
    print("\n1. Создание датасета...")
    try:
        dataset = LatentPromptDataset(latents_dir, prompts_dir)
        print("✅ Датасет создан успешно")
    except Exception as e:
        print(f"❌ Ошибка создания датасета: {e}")
        exit(1)
    
    print(f"\n2. Загрузка первого сэмпла...")
    try:
        latent, prompt = dataset[0]
        print(f"   Latent shape: {latent.shape}")
        print(f"   Latent dtype: {latent.dtype}")
        print(f"   Prompt: '{prompt}'")
        print("✅ Первый сэмпл загружен")
    except Exception as e:
        print(f"❌ Ошибка загрузки сэмпла: {e}")
        exit(1)
    
    print(f"\n3. Тестирование нескольких сэмплов...")
    try:
        for i in range(min(3, len(dataset))):
            latent, prompt = dataset[i]
            print(f"   Сэмпл {i}: latent shape={latent.shape}, prompt='{prompt[:30]}...'")
        print("✅ Множественные сэмплы загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки множественных сэмплов: {e}")
        exit(1)
    
    print(f"\n4. Тестирование DataLoader...")
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False, 
            num_workers=0,
            collate_fn=custom_collate
        )
        batch_latents, batch_prompts = next(iter(dataloader))
        print(f"   Batch latents shape: {batch_latents.shape}")
        print(f"   Batch size: {len(batch_prompts)}")
        print(f"   First prompt: '{batch_prompts[0]}'")
        print("✅ DataLoader работает")
    except Exception as e:
        print(f"❌ Ошибка DataLoader: {e}")
        exit(1)
    
    print("\n" + "="*60)
    print("✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
    print("Данные готовы для обучения.")
    print("="*60)



