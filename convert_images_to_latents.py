import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion

def convert_images_to_latents(images_dir, latents_dir, prompts_dir, output_latents_dir, output_prompts_dir):
    """Конвертируем изображения в латенты используя VAE из micro_diffusion"""
    print(f"Конвертируем изображения из {images_dir} в латенты...")
    
    # Создаем выходные директории
    os.makedirs(output_latents_dir, exist_ok=True)
    os.makedirs(output_prompts_dir, exist_ok=True)
    
    # Загружаем модель для VAE
    print("Загружаем модель для VAE...")
    model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cpu")
    
    # Загружаем веса
    try:
        weights = torch.load("./micro_diffusion/trained_models/teacher.pt", map_location="cpu")
        model.dit.load_state_dict(weights, strict=False)
        model.eval()
        print("Модель загружена успешно")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return
    
    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Стандартный размер для VAE
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация в [-1, 1]
    ])
    
    # Получаем список файлов
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
    
    print(f"Найдено {len(image_files)} изображений и {len(prompt_files)} промптов")
    
    good_count = 0
    error_count = 0
    
    for i, image_file in enumerate(image_files):
        try:
            # Загружаем изображение
            image_path = os.path.join(images_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            
            # Применяем трансформации
            image_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]
            
            # Кодируем в латентное пространство через VAE
            with torch.no_grad():
                # Используем VAE для кодирования
                latent = model.vae.encode(image_tensor).latent_dist.sample()
                # Масштабируем согласно VAE scaling factor
                latent = latent * model.vae.config.scaling_factor
            
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
            prompt_path = os.path.join(prompts_dir, prompt_filename)
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
    prompts_dir = r"C:\newTry2\train\dataset_stub_4k"  # Промпты в той же папке
    output_latents_dir = r"C:\newTry2\train\datadir\latents_good"
    output_prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    # Конвертируем
    convert_images_to_latents(images_dir, images_dir, prompts_dir, output_latents_dir, output_prompts_dir)
