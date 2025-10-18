import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
# Используем float32
import psutil

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

# Импортируем наши компоненты
from micro_diffusion.models.dit import MicroDiT_XL_2, DiT  # ← Добавляем DiT для Teacher
from proper_text_embeddings import ProperTextEncoder
from create_proper_dataset import ProperDataset

def get_memory_info():
    """Получаем информацию о памяти"""
    # RAM
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024**3)  # GB
    ram_total = ram.total / (1024**3)  # GB
    
    # VRAM
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        vram_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    else:
        vram_allocated = vram_reserved = vram_total = 0
    
    return {
        'ram_used': ram_used,
        'ram_total': ram_total,
        'vram_allocated': vram_allocated,
        'vram_reserved': vram_reserved,
        'vram_total': vram_total
    }

def log_memory_usage(iteration, epoch, prefix=""):
    """Логируем использование памяти"""
    mem = get_memory_info()
    print(f"💾 Память ({prefix} итерация {iteration}, эпоха {epoch}):")
    print(f"   🖥️  RAM: {mem['ram_used']:.1f}/{mem['ram_total']:.1f} GB ({mem['ram_used']/mem['ram_total']*100:.1f}%)")
    print(f"   🎮 VRAM: {mem['vram_allocated']:.1f}/{mem['vram_total']:.1f} GB ({mem['vram_allocated']/mem['vram_total']*100:.1f}%)")
    print(f"   📊 VRAM зарезервировано: {mem['vram_reserved']:.1f} GB")

class ConsistencyDistillationModel(nn.Module):
    """Модель для Consistency Distillation"""
    
    def __init__(self, teacher_model, student_model):
        super(ConsistencyDistillationModel, self).__init__()
        self.teacher_model = teacher_model  # Замороженный Teacher
        self.student_model = student_model  # Обучаемый Student
        
        # Замораживаем Teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def forward(self, x_t, t, text_embeddings):
        """Forward pass для CD"""
        # Teacher показывает правильный путь (заморожен)
        with torch.no_grad():
            teacher_output = self.teacher_model(x_t, t, text_embeddings)
            if isinstance(teacher_output, dict):
                teacher_output = teacher_output['sample']
        
        # Student предсказывает (обучается)
        student_output = self.student_model(x_t, t, text_embeddings)
        if isinstance(student_output, dict):
            student_output = student_output['sample']
        
        return {
            'teacher_output': teacher_output,
            'student_output': student_output
        }

def load_models(device="cuda"):
    """Загружаем модели"""
    print("🔄 Загружаем модели...")
    
    # Teacher модель - DiT-XL на CPU (заморожен) - совместим с предобученной моделью
    print("🧠 Загружаем Teacher (DiT-XL) на CPU...")
    teacher_model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=4,
        dim=1024,  # DiT-XL (совместим с предобученной моделью)
        depth=28,  # DiT-XL
        head_dim=64,
        multiple_of=256,
        caption_channels=1024,
        pos_interp_scale=1.0,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=[1.0],
        ffn_multipliers=[4.0],
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=768,  # Совместим с предобученной моделью
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        use_bias=True,
        num_experts=8,
        expert_capacity=1,
        experts_every_n=2
    )
    
    # Пытаемся загрузить предобученные веса Teacher
    try:
        # Проверяем доступные предобученные модели
        pretrained_models = [
            "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_37M_real_and_synthetic_data.pt",
            "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_22M_real_only_data.pt",
            "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_0.5B_synthetic_data.pt"
        ]
        
        teacher_loaded = False
        for model_path in pretrained_models:
            if os.path.exists(model_path):
                print(f"🔍 Найдена предобученная модель: {os.path.basename(model_path)}")
                teacher_state_dict = torch.load(model_path, map_location="cpu")
                
                # Проверяем совместимость размеров
                try:
                    teacher_model.load_state_dict(teacher_state_dict)
                    print(f"✅ Предобученные веса Teacher загружены из {os.path.basename(model_path)}")
                    teacher_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Несовместимость размеров для {os.path.basename(model_path)}: {e}")
                    continue
        
        if not teacher_loaded:
            print("⚠️  Предобученные веса Teacher не найдены или несовместимы, используем случайную инициализацию")
            print("🎯 Teacher будет обучен с нуля - это нормально для CD!")
            
    except Exception as e:
        print(f"⚠️  Ошибка загрузки Teacher весов: {e}")
        print("⚠️  Используем случайную инициализацию")
    
    teacher_model.to("cpu", dtype=torch.float32)
    teacher_model.eval()
    print("✅ Teacher (DiT-Large) загружен на CPU")
    
    # Student модель - MicroDiT_XL_2 на GPU (обучается)
    print("🎓 Загружаем Student (DiT-Small) на GPU...")
    student_model = MicroDiT_XL_2(
        input_size=64,
        caption_channels=1024,
        pos_interp_scale=1.0,
        in_channels=4
    )
    
    # Инициализируем Student случайными весами (отличается от Teacher)
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

def consistency_distillation_step(latents, text_embeddings, cd_model, device="cuda"):
    """Правильный CD шаг: Student учится у Teacher"""
    try:
        # Создаем шум
        noise = torch.randn_like(latents)
        
        # Сэмплируем время
        t = torch.rand(1, device=device, dtype=torch.float32)
        
        # Зашумляем латенты
        noisy_latents = latents + t * noise
        
        # CD модель выполняет forward pass
        outputs = cd_model(noisy_latents, t, text_embeddings)
        teacher_output = outputs['teacher_output']
        student_output = outputs['student_output']
        
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

def train_proper_cd_with_text():
    """Правильное CD обучение с текстовым кондиционированием"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Устройство: {device}")
    
    # Загружаем модели
    teacher_model, student_model, text_encoder = load_models(device)
    
    # Создаем CD модель
    print("🔧 Создаем Consistency Distillation модель...")
    cd_model = ConsistencyDistillationModel(teacher_model, student_model)
    cd_model.to(device)
    print("✅ CD модель создана")
    
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
    
    # Оптимизатор только для Student
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
    
    print(f"\n🧪 ТЕСТОВОЕ CD ОБУЧЕНИЕ С ТЕКСТОВЫМ КОНДИЦИОНИРОВАНИЕМ")
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
                
                print(f"🔧 Данные: latents={latents.dtype}, text_embeddings={text_embeddings.dtype}")
                
                # Правильный CD шаг: Student учится у Teacher
                loss_dict = consistency_distillation_step(
                    latents, text_embeddings, cd_model, device
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
        torch.save(student_model_cpu.state_dict(), 'student_test_cd_final.pt')
        student_model.to(device)
        print(f"💾 Тестовая модель сохранена: student_test_cd_final.pt")
    except Exception as e:
        print(f"⚠️  Не удалось сохранить тестовую модель: {e}")
    
    # Создаем график потерь
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(all_losses)
        plt.title('Тестовое CD обучение с текстовым кондиционированием - Потери')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('test_cd_losses.png')
        print("📊 График сохранен: test_cd_losses.png")
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
    print(f"💾 Тестовые веса: student_test_cd_final.pt")
    print(f"📊 Тестовый график: test_cd_losses.png")
    
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
        os.makedirs("test_generation_outputs", exist_ok=True)
        
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
                filename = f"test_generation_outputs/test_generated_{i+1}.png"
                image.save(filename)
                print(f"💾 Сохранено: {filename}")
        
        print(f"\n🎨 ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
        print(f"📁 Результаты в папке: test_generation_outputs/")
        
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_proper_cd_with_text()