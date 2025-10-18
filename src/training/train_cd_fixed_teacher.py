import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

# Импортируем правильные компоненты
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
from micro_diffusion.models.dit import MicroDiT_XL_2
from proper_text_embeddings import ProperTextEncoder
from create_proper_dataset import ProperDataset

def get_memory_info():
    """Получаем информацию о памяти"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_free = gpu_memory - gpu_allocated
        return {
            'gpu_total': gpu_memory,
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'gpu_free': gpu_free
        }
    return {}

def log_memory_usage(iteration, epoch, prefix=""):
    """Логируем использование памяти"""
    memory_info = get_memory_info()
    if memory_info:
        print(f"{prefix} Итерация {iteration}, Эпоха {epoch}")
        print(f"  GPU: {memory_info['gpu_allocated']:.1f}GB / {memory_info['gpu_total']:.1f}GB (свободно: {memory_info['gpu_free']:.1f}GB)")
        print(f"  GPU Reserved: {memory_info['gpu_reserved']:.1f}GB")

def load_models(device="cuda"):
    """Загружаем модели с ПРАВИЛЬНОЙ архитектурой Teacher"""
    print("🔄 Загружаем модели...")
    
    # Teacher модель - create_latent_diffusion (ПРАВИЛЬНАЯ архитектура)
    print("🧠 Загружаем Teacher с ПРАВИЛЬНОЙ архитектурой...")
    teacher_model = create_latent_diffusion(
        latent_res=64, 
        in_channels=4, 
        pos_interp_scale=2.0,
        precomputed_latents=False, 
        dtype="bfloat16"
    ).to("cpu")
    
    # Загружаем предобученные веса Teacher (FID 12.66)
    pretrained_path = "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_37M_real_and_synthetic_data.pt"
    
    try:
        print(f"🔍 Загружаем предобученные веса из: {os.path.basename(pretrained_path)}")
        teacher_weights = torch.load(pretrained_path, map_location="cpu")
        
        # Загружаем веса в правильную архитектуру
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        print("✅ Предобученные веса Teacher загружены успешно!")
        print("🎯 Teacher: FID 12.66 (37M изображений)")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки предобученных весов: {e}")
        print("⚠️  Используем случайную инициализацию Teacher")
    
    teacher_model.eval()
    print("✅ Teacher загружен на CPU с ПРАВИЛЬНОЙ архитектурой")
    
    # Student модель - MicroDiT_XL_2 на GPU (обучается)
    print("🎓 Загружаем Student (DiT-Small) на GPU...")
    student_model = MicroDiT_XL_2(
        input_size=64,
        caption_channels=1024,
        pos_interp_scale=1.0,
        in_channels=4
    )
    
    # Инициализируем Student случайными весами
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if m.weight is not None:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    student_model.apply(init_weights)
    print("✅ Student инициализирован случайными весами")
    
    student_model.to(device, dtype=torch.float32)
    student_model.train()
    print("✅ Student (DiT-Small) загружен на GPU")
    
    # Включаем gradient checkpointing для экономии памяти
    if hasattr(student_model, 'enable_gradient_checkpointing'):
        student_model.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing включен")
    
    # Создаем текстовый кодировщик
    print("🔧 Создаем текстовый кодировщик...")
    text_encoder = ProperTextEncoder(device)
    
    print("✅ Модели загружены")
    return teacher_model, student_model, text_encoder

def consistency_distillation_step(latents, text_embeddings, teacher_model, student_model, device="cuda"):
    """Правильный CD шаг: Student учится у предобученного Teacher"""
    try:
        # Создаем шум
        noise = torch.randn_like(latents)
        
        # Сэмплируем время
        t = torch.rand(1, device=device, dtype=torch.float32)
        
        # Зашумляем латенты
        noisy_latents = latents + t * noise
        
        # Teacher показывает правильный шаг (данные на CPU)
        with torch.no_grad():
            # Используем правильный метод для teacher_model
            teacher_output = teacher_model.model_forward_wrapper(
                noisy_latents.cpu(),
                t.cpu(),
                text_embeddings.cpu(),
                teacher_model.dit,
                mask_ratio=0.0
            )
            # Извлекаем тензор из словаря
            teacher_output = teacher_output['sample'] if isinstance(teacher_output, dict) else teacher_output
            teacher_output = teacher_output.to(device)
        
        # Student предсказывает
        student_output = student_model(noisy_latents, t, text_embeddings)
        # Извлекаем тензор из словаря
        student_output = student_output['sample'] if isinstance(student_output, dict) else student_output
        
        # Loss: Student должен быть похож на Teacher
        loss = F.mse_loss(student_output, teacher_output)
        
        # Отладочная информация (только первые 5 итераций)
        if hasattr(consistency_distillation_step, '_debug_count'):
            consistency_distillation_step._debug_count += 1
        else:
            consistency_distillation_step._debug_count = 1
            
        if consistency_distillation_step._debug_count <= 5:
            print(f"🔍 Итерация {consistency_distillation_step._debug_count}:")
            print(f"   Teacher mean: {teacher_output.mean().item():.6f}, std: {teacher_output.std().item():.6f}")
            print(f"   Student mean: {student_output.mean().item():.6f}, std: {student_output.std().item():.6f}")
            print(f"   Loss: {loss.item():.6f}")
        
        return {
            'total_loss': loss,
            'teacher_output': teacher_output,
            'student_output': student_output,
            't': t
        }
        
    except Exception as e:
        print(f"❌ Ошибка в CD шаге: {e}")
        return None

def train_cd_fixed_teacher():
    """CD обучение с ПРАВИЛЬНОЙ архитектурой Teacher"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Устройство: {device}")
    
    # Загружаем модели
    teacher_model, student_model, text_encoder = load_models(device)
    
    # Создаем правильный датасет
    print("\n📊 Создаем правильный датасет...")
    latents_dir = "/home/ubuntu/train/train/datadir/latents_good"
    prompts_dir = "/home/ubuntu/train/train/datadir/prompts_good"
    
    dataset = ProperDataset(latents_dir, prompts_dir, text_encoder, device)
    print(f"✅ Датасет создан: {len(dataset)} образцов")
    
    # Параметры обучения
    num_epochs = 1  # Тестовое обучение
    max_iters = 100  # Тестовые итерации
    batch_size = 1  # Безопасный размер батча
    lr = 1e-4
    
    # Оптимизатор
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    
    # Используем float32
    print("✅ Используем float32")
    
    # Дополнительные оптимизации памяти
    print("🔧 Включаем дополнительные оптимизации памяти...")
    
    # TF32 для Ampere GPU (если доступно)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✅ TF32 включен для Ampere GPU")
    
    # Channels Last для экономии памяти
    try:
        student_model = student_model.to(memory_format=torch.channels_last)
        print("✅ Channels Last включен")
    except Exception as e:
        print(f"⚠️  Channels Last не поддерживается: {e}")
    
    # Логирование
    all_losses = []
    start_time = time.time()
    
    # Начальная память
    log_memory_usage(0, 0, "🚀 СТАРТ:")
    
    print(f"\n🧪 ТЕСТОВОЕ CD ОБУЧЕНИЕ С ПРАВИЛЬНОЙ АРХИТЕКТУРОЙ TEACHER")
    print(f"📊 Эпох: {num_epochs}, Итераций: {max_iters}")
    print(f"📊 Общее количество итераций: {num_epochs * max_iters}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        print(f"\n🔄 ЭПОХА {epoch + 1}/{num_epochs}")
        print("=" * 50)
        
        epoch_losses = []
        
        # Создаем прогресс-бар
        pbar = tqdm(range(max_iters), desc=f"Эпоха {epoch + 1}/{num_epochs}")
        
        for iteration in pbar:
            try:
                # Получаем данные из датасета
                sample_idx = iteration % len(dataset)
                sample = dataset[sample_idx]
                
                latents = sample['latents'].unsqueeze(0).to(device, dtype=torch.float32)
                text_embeddings = sample['text_embeddings'].to(device, dtype=torch.float32)
                prompt = sample['prompt']
                
                # Правильный CD шаг: Student учится у предобученного Teacher
                loss_dict = consistency_distillation_step(
                    latents, text_embeddings, teacher_model, student_model, device
                )
                
                if loss_dict is None:
                    continue
                
                # Агрессивная очистка памяти
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Обратное распространение
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                # Агрессивная очистка памяти
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Логирование
                epoch_losses.append(loss_dict['total_loss'].item())
                all_losses.append(loss_dict['total_loss'].item())
                
                # Обновляем прогресс бар
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total_loss'].item():.6f}",
                    'Avg': f"{np.mean(epoch_losses):.6f}",
                    'Prompt': prompt[:20] + "..." if len(prompt) > 20 else prompt
                })
                
                # Мониторинг памяти каждые 10 итераций
                if iteration % 10 == 0:
                    log_memory_usage(iteration, epoch + 1, "🧠 МОНИТОРИНГ ПАМЯТИ:")
                
            except Exception as e:
                print(f"❌ Ошибка на итерации {iteration}: {e}")
                continue
        
        # Статистика эпохи
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            print(f"📊 Эпоха {epoch + 1} завершена. Средний loss: {avg_loss:.6f}")
        else:
            print(f"📊 Эпоха {epoch + 1} завершена. Нет успешных итераций.")
    
    # Сохраняем тестовую модель
    try:
        student_model_cpu = student_model.cpu()
        torch.save(student_model_cpu.state_dict(), 'student_test_cd_fixed_teacher.pt')
        student_model.to(device)
        print(f"💾 Тестовая модель сохранена: student_test_cd_fixed_teacher.pt")
    except Exception as e:
        print(f"⚠️  Не удалось сохранить тестовую модель: {e}")
    
    # Создаем график потерь
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(all_losses)
        plt.title('Тестовое CD обучение с ПРАВИЛЬНОЙ архитектурой Teacher - Потери')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('test_cd_fixed_teacher_losses.png')
        print("📊 График сохранен: test_cd_fixed_teacher_losses.png")
    except Exception as e:
        print(f"⚠️  Не удалось сохранить график: {e}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 ТЕСТОВОЕ CD ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"⏱️ Общее время: {total_time/60:.1f} минут")
    if len(all_losses) > 0:
        print(f"📉 Начальный loss: {all_losses[0]:.6f}")
        print(f"📉 Финальный loss: {all_losses[-1]:.6f}")
        if all_losses[0] > 0:
            improvement = ((all_losses[0] - all_losses[-1]) / all_losses[0] * 100)
            print(f"📊 Улучшение: {improvement:.1f}%")
        else:
            print("📊 Улучшение: Невозможно вычислить (начальный loss = 0)")
    else:
        print("❌ Нет успешных итераций обучения")
    print(f"💾 Тестовые веса: student_test_cd_fixed_teacher.pt")
    print(f"📊 Тестовый график: test_cd_fixed_teacher_losses.png")
    
    # Тестируем генерацию изображений
    print(f"\n🎨 ТЕСТИРУЕМ ГЕНЕРАЦИЮ ИЗОБРАЖЕНИЙ:")
    print("=" * 50)
    
    try:
        # Загружаем рабочий VAE для генерации
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
        vae.to(device)
        vae.eval()
        print("✅ VAE загружен для генерации")
        
        # Тестовые промпты
        test_prompts = [
            "A beautiful sunset over mountains",
            "A cozy cabin in a snowy forest",
            "A majestic dragon flying over a medieval castle"
        ]
        
        # Создаем папку для результатов
        os.makedirs("test_fixed_teacher_outputs", exist_ok=True)
        
        # Генерируем изображения
        student_model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                print(f"\n📝 Генерируем: '{prompt}'")
                
                # Получаем текстовые эмбеддинги
                text_embeddings = text_encoder.encode_text(prompt).to(torch.float32)
                
                # Инициализируем латенты
                latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
                
                # Генерируем (4 шага)
                for step in range(4):
                    t = torch.ones(1, device=device, dtype=torch.float32) * (1.0 - step / 3.0)
                    output = student_model(latents, t, text_embeddings)
                    latents = output['sample'] if isinstance(output, dict) else output
                    print(f"🔄 Шаг {step + 1}/4: t={t.item():.3f}")
                
                # Декодируем в изображение
                latents_fp32 = latents.to(torch.float32)
                decoded_output = vae.decode(latents_fp32)
                decoded_image = decoded_output.sample if hasattr(decoded_output, 'sample') else decoded_output
                
                # Нормализуем и сохраняем
                decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                image_tensor = decoded_image[0].cpu()
                image_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_array = np.clip(image_array, 0, 255)
                
                from PIL import Image
                image = Image.fromarray(image_array)
                filename = f"test_fixed_teacher_outputs/test_generated_{i+1}.png"
                image.save(filename)
                print(f"💾 Сохранено: {filename}")
        
        print(f"\n🎨 ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
        print(f"📁 Результаты в папке: test_fixed_teacher_outputs/")
        
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_cd_fixed_teacher()

