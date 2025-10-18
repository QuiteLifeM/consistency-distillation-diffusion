"""
БЫСТРАЯ конвертация изображений в латенты с batch processing
"""
import os
import torch
from PIL import Image
from tqdm import tqdm
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
import torchvision.transforms as transforms

def convert_images_to_latents_fast(
    images_dir,
    output_latents_dir="datadir/latents_good",
    output_prompts_dir="datadir/prompts_good",
    batch_size=16  # VAE encoder может обрабатывать больше изображений
):
    """
    БЫСТРАЯ конвертация изображений в латенты с batch processing
    """
    print("🔄 БЫСТРАЯ конвертация изображений в латенты")
    print("=" * 70)
    print(f"⚡ Batch size: {batch_size}")
    
    # Проверяем CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, используем CPU (будет медленно)")
        device = "cpu"
        batch_size = 4  # CPU меньше batch
    else:
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    
    # Создаем директории
    os.makedirs(output_latents_dir, exist_ok=True)
    os.makedirs(output_prompts_dir, exist_ok=True)
    print(f"📁 Латенты: {output_latents_dir}/")
    print(f"📁 Промпты: {output_prompts_dir}/")
    
    # Загружаем модель (нужен только VAE)
    print("\n📥 Загрузка VAE из micro_diffusion...")
    model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="bfloat16"
    ).to(device)
    model.eval()
    print("✅ VAE загружен")
    
    # Преобразование изображений
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Получаем список изображений
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"\n📊 Найдено {len(image_files)} изображений")
    
    print("\n🔄 Конвертация в латенты (батчами)...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    # Обрабатываем батчами
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    import time
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Батчи"):
        try:
            # Определяем изображения для текущего батча
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            # Загружаем и преобразуем все изображения батча
            batch_images = []
            batch_prompts = []
            
            for img_file in batch_files:
                img_path = os.path.join(images_dir, img_file)
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                
                # Читаем промпт
                txt_file = img_file.replace('.png', '.txt')
                txt_path = os.path.join(images_dir, txt_file)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        batch_prompts.append(f.read().strip())
                else:
                    batch_prompts.append("")
            
            # Объединяем в батч
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Кодируем ВСЕ изображения батча ПАРАЛЛЕЛЬНО через VAE!
            with torch.no_grad():
                latents = model.vae.encode(batch_tensor.to(torch.bfloat16)).latent_dist.sample()
                latents = latents * 0.13025
            
            # Сохраняем все латенты и промпты из батча
            for i, (img_file, prompt) in enumerate(zip(batch_files, batch_prompts)):
                # Сохраняем латент (БЕЗ лишней batch dimension)
                latent = latents[i]  # [4, 64, 64] вместо [1, 4, 64, 64]
                latent_filename = img_file.replace('.png', '.pt')
                latent_path = os.path.join(output_latents_dir, latent_filename)
                torch.save(latent.cpu(), latent_path)
                
                # Сохраняем промпт
                txt_filename = img_file.replace('.png', '.txt')
                dst_txt_path = os.path.join(output_prompts_dir, txt_filename)
                with open(dst_txt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                successful += 1
            
            # Очищаем память каждые 10 батчей
            if device == "cuda" and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n⚠️ Ошибка в батче {batch_idx}: {e}")
            failed += len(batch_files)
            continue
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
    print(f"📊 Статистика:")
    print(f"   ✅ Успешно: {successful}")
    print(f"   ❌ Ошибок: {failed}")
    print(f"   ⏱️  Время: {elapsed_time/60:.1f} минут ({elapsed_time/successful:.2f} сек/изображение)")
    print(f"   🚀 Скорость: {successful/(elapsed_time/60):.1f} изображений/минуту")
    print(f"   📁 Латенты: {output_latents_dir}/")
    print(f"   📁 Промпты: {output_prompts_dir}/")
    print("=" * 70)
    
    # Проверяем размеры латентов
    if successful > 0:
        sample_latent_path = os.path.join(output_latents_dir, sorted(os.listdir(output_latents_dir))[0])
        sample_latent = torch.load(sample_latent_path)
        print(f"\n📊 Размер латентов: {sample_latent.shape}")
        print(f"   Ожидается: torch.Size([1, 4, 64, 64])")
    
    print("\n💡 Следующий шаг:")
    print("   python3 train_true_consistency_distillation.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='БЫСТРАЯ конвертация изображений в латенты')
    parser.add_argument('--images_dir', type=str, default='dataset_sdxl_turbo', 
                        help='Директория с изображениями')
    parser.add_argument('--output_latents_dir', type=str, default='datadir/latents_good',
                        help='Выходная директория для латентов')
    parser.add_argument('--output_prompts_dir', type=str, default='datadir/prompts_good',
                        help='Выходная директория для промптов')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16 для RTX 3090)')
    
    args = parser.parse_args()
    
    convert_images_to_latents_fast(
        images_dir=args.images_dir,
        output_latents_dir=args.output_latents_dir,
        output_prompts_dir=args.output_prompts_dir,
        batch_size=args.batch_size
    )

