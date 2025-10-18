import os
import torch
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
from PIL import Image
import numpy as np

# Настройка
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def get_text_embeddings(prompts, model, device="cuda"):
    """Получение текстовых эмбеддингов"""
    tokenized = model.tokenizer.tokenize(prompts)
    input_ids = tokenized['input_ids'].to(device)
    
    with torch.no_grad():
        text_embeddings = model.text_encoder.encode(input_ids)[0]
    
    return text_embeddings

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 ДИАГНОСТИКА ГЕНЕРАЦИИ")
    print("=" * 80)
    
    # Проверяем CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна!")
        exit(1)
    
    print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    
    # Путь к весам учителя
    teacher_weights_path = "./micro_diffusion/micro_diffusion/trained_models/teacher.pt"
    
    # Создаем модель учителя на GPU
    print("\n📥 Загрузка модели учителя на GPU (Float16)...")
    teacher_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float16"
    ).to("cuda")
    
    # Загружаем оригинальные веса учителя
    print("📥 Загрузка весов учителя...")
    teacher_weights = torch.load(teacher_weights_path, map_location="cuda")
    teacher_model.dit.load_state_dict(teacher_weights, strict=False)
    teacher_model.eval()
    print("✅ Модель учителя загружена")
    
    # Проверяем веса модели
    print("\n🔍 Проверка весов модели:")
    has_nan = False
    for name, param in teacher_model.dit.named_parameters():
        if torch.isnan(param).any():
            print(f"   ❌ NaN в {name}")
            has_nan = True
    
    if not has_nan:
        print("   ✅ Веса модели в порядке (нет NaN)")
    
    # Тестовый промпт
    prompt = "A beautiful sunset over mountains"
    
    print(f"\n🎨 Генерация: '{prompt}'")
    
    with torch.no_grad():
        # 1. Текстовые эмбеддинги
        print("\n1️⃣ Получение текстовых эмбеддингов...")
        text_embeddings = get_text_embeddings([prompt], teacher_model, device="cuda")
        print(f"   Shape: {text_embeddings.shape}")
        print(f"   Range: [{text_embeddings.min():.3f}, {text_embeddings.max():.3f}]")
        print(f"   Has NaN: {torch.isnan(text_embeddings).any()}")
        
        # 2. Начальный шум
        print("\n2️⃣ Создание начального шума...")
        latent_shape = (1, 4, 64, 64)
        latents = torch.randn(latent_shape, device="cuda")
        print(f"   Shape: {latents.shape}")
        print(f"   Range: [{latents.min():.3f}, {latents.max():.3f}]")
        
        # 3. EDM параметры
        print("\n3️⃣ EDM параметры:")
        sigma_min = teacher_model.edm_config.sigma_min
        sigma_max = teacher_model.edm_config.sigma_max
        rho = teacher_model.edm_config.rho
        print(f"   sigma_min: {sigma_min}")
        print(f"   sigma_max: {sigma_max}")
        print(f"   rho: {rho}")
        
        # 4. Денойзинг (упрощенный - 4 шага)
        print("\n4️⃣ Денойзинг (4 шага)...")
        num_steps = 4
        step_indices = torch.arange(num_steps, device="cuda")
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        print(f"   T steps: {t_steps}")
        
        x = latents * t_steps[0]
        print(f"   Initial noisy x: [{x.min():.3f}, {x.max():.3f}]")
        
        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            
            sigma_batch = torch.full((1,), t_cur, device="cuda")
            
            output = teacher_model.model_forward_wrapper(
                x,
                sigma_batch,
                text_embeddings,
                teacher_model.dit,
                mask_ratio=0.0
            )
            
            denoised = output['sample']
            
            print(f"   Step {i+1}: t_cur={t_cur:.3f}, denoised range=[{denoised.min():.3f}, {denoised.max():.3f}], has_nan={torch.isnan(denoised).any()}")
            
            d = (x - denoised) / t_cur if t_cur > 0 else torch.zeros_like(x)
            x = x + d * (t_next - t_cur)
            
            print(f"           new x range=[{x.min():.3f}, {x.max():.3f}], has_nan={torch.isnan(x).any()}")
        
        print(f"\n   Final latents: [{x.min():.3f}, {x.max():.3f}]")
        print(f"   Has NaN: {torch.isnan(x).any()}")
        print(f"   Has Inf: {torch.isinf(x).any()}")
        
        # 5. Декодирование через VAE
        print("\n5️⃣ Декодирование через VAE...")
        
        # Проверяем веса VAE
        print("   Проверка весов VAE decoder:")
        vae_has_nan = False
        for name, param in teacher_model.vae.decoder.named_parameters():
            if torch.isnan(param).any():
                print(f"      ❌ NaN в {name}")
                vae_has_nan = True
        
        if not vae_has_nan:
            print("      ✅ Веса VAE в порядке")
        
        # Декодируем
        x_scaled = x / 0.13025
        print(f"   Scaled latents: [{x_scaled.min():.3f}, {x_scaled.max():.3f}]")
        print(f"   Dtype: {x_scaled.dtype}")
        
        # Приводим к float16
        x_scaled = x_scaled.to(torch.float16)
        print(f"   After to(float16): [{x_scaled.min():.3f}, {x_scaled.max():.3f}]")
        
        images = teacher_model.vae.decode(x_scaled).sample
        print(f"   Decoded images: {images.shape}")
        print(f"   Range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Has NaN: {torch.isnan(images).any()}")
        print(f"   Has Inf: {torch.isinf(images).any()}")
        
        # 6. Конвертация в изображение
        print("\n6️⃣ Конвертация в изображение...")
        
        # Нормализация
        images_norm = (images / 2 + 0.5).clamp(0, 1)
        print(f"   After normalization: [{images_norm.min():.3f}, {images_norm.max():.3f}]")
        
        # В numpy
        images_np = images_norm.cpu().permute(0, 2, 3, 1).numpy()
        print(f"   Numpy shape: {images_np.shape}")
        print(f"   Numpy range: [{images_np.min():.3f}, {images_np.max():.3f}]")
        
        # В uint8
        images_uint8 = (images_np * 255).round().astype("uint8")
        print(f"   Uint8 range: [{images_uint8.min()}, {images_uint8.max()}]")
        print(f"   Unique values: {len(np.unique(images_uint8))}")
        
        # Сохраняем
        os.makedirs("debug_outputs", exist_ok=True)
        
        # Вариант 1: Стандартный
        img1 = Image.fromarray(images_uint8[0])
        img1.save("debug_outputs/debug_standard.png")
        print(f"   ✅ Сохранено: debug_outputs/debug_standard.png")
        
        # Вариант 2: MinMax нормализация
        images_minmax = images_np[0]
        images_minmax = (images_minmax - images_minmax.min()) / (images_minmax.max() - images_minmax.min() + 1e-8)
        images_minmax = (images_minmax * 255).round().astype("uint8")
        img2 = Image.fromarray(images_minmax)
        img2.save("debug_outputs/debug_minmax.png")
        print(f"   ✅ Сохранено: debug_outputs/debug_minmax.png")
        
        # Вариант 3: Без нормализации /2 + 0.5
        images_raw = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images_raw = (images_raw * 255).round().astype("uint8")
        img3 = Image.fromarray(images_raw[0])
        img3.save("debug_outputs/debug_raw.png")
        print(f"   ✅ Сохранено: debug_outputs/debug_raw.png")
    
    print("\n" + "=" * 80)
    print("🎉 ДИАГНОСТИКА ЗАВЕРШЕНА!")
    print("=" * 80)
    print("📁 Проверьте папку debug_outputs/")
    print("   - debug_standard.png (стандартная нормализация)")
    print("   - debug_minmax.png (MinMax нормализация)")
    print("   - debug_raw.png (без нормализации)")
    print("=" * 80)










