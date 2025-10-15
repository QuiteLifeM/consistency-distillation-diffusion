import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def convert_images_to_latents_simple(images_dir, output_latents_dir, output_prompts_dir):
    """Простая конвертация изображений в латенты (без VAE)"""
    print(f"Конвертируем изображения из {images_dir} в латенты...")
    
    # Создаем выходные директории
    os.makedirs(output_latents_dir, exist_ok=True)
    os.makedirs(output_prompts_dir, exist_ok=True)
    
    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Размер латента
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация в [-1, 1]
    ])
    
    # Получаем список файлов
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    print(f"Найдено {len(image_files)} изображений")
    
    good_count = 0
    error_count = 0
    
    for i, image_file in enumerate(image_files):
        try:
            # Загружаем изображение
            image_path = os.path.join(images_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            
            # Применяем трансформации
            image_tensor = transform(image)  # [3, 64, 64]
            
            # Создаем латент [4, 64, 64] - добавляем один канал
            # Простое решение: дублируем один канал
            latent = torch.cat([image_tensor, image_tensor[0:1]], dim=0)  # [4, 64, 64]
            
            # Проверяем на NaN
            if torch.isnan(latent).any():
                print(f"NaN в латенте {image_file}, пропускаем...")
                error_count += 1
                continue
            
            # Сохраняем латент
            latent_filename = image_file.replace('.png', '.pt')
            latent_path = os.path.join(output_latents_dir, latent_filename)
            torch.save(latent, latent_path)
            
            # Копируем промпт
            prompt_filename = image_file.replace('.png', '.txt')
            prompt_path = os.path.join(images_dir, prompt_filename)
            output_prompt_path = os.path.join(output_prompts_dir, prompt_filename)
            
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                with open(output_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            
            good_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Обработано {i+1}/{len(image_files)} изображений")
                
        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {e}")
            error_count += 1
            continue
    
    print(f"\nРезультаты:")
    print(f"Успешно обработано: {good_count}")
    print(f"Ошибок: {error_count}")
    print(f"Латенты сохранены в: {output_latents_dir}")
    print(f"Промпты сохранены в: {output_prompts_dir}")

if __name__ == "__main__":
    # Пути к данным
    images_dir = r"C:\newTry2\train\dataset_stub_4k"
    output_latents_dir = r"C:\newTry2\train\datadir\latents_good"
    output_prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    # Конвертируем
    convert_images_to_latents_simple(images_dir, output_latents_dir, output_prompts_dir)

