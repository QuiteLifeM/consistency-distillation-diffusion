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

from micro_diffusion.models.dit import MicroDiT_XL_2, DiT
from proper_text_embeddings import ProperTextEncoder
from create_proper_dataset import ProperDataset

def get_memory_info():
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
    memory_info = get_memory_info()
    if memory_info:
        print(f"{prefix} Итерация {iteration}, Эпоха {epoch}")
        print(f"  GPU: {memory_info['gpu_allocated']:.1f}GB / {memory_info['gpu_total']:.1f}GB (свободно: {memory_info['gpu_free']:.1f}GB)")
        print(f"  GPU Reserved: {memory_info['gpu_reserved']:.1f}GB")

def load_models(device="cuda"):
    print("")
    
    print("")
    teacher_model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=4,
        dim=1024,
        depth=28,
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
        patch_mixer_dim=768,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=1,
        experts_every_n=2
    )
    
    pretrained_path = "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_37M_real_and_synthetic_data.pt"
    
    try:
        print(f" Загружаем предобученные веса из: {os.path.basename(pretrained_path)}")
        teacher_state_dict = torch.load(pretrained_path, map_location="cpu")
        
        teacher_model.load_state_dict(teacher_state_dict)
        print(" Предобученные веса Teacher загружены успешно!")
        
    except Exception as e:
        print(f" Ошибка загрузки предобученных весов: {e}")
        print("  Используем случайную инициализацию Teacher")
    
    teacher_model.to("cpu", dtype=torch.float32)
    teacher_model.eval()
    print(" Teacher (DiT-XL) загружен на CPU с предобученными весами")
    
    print(" Загружаем Student (DiT-Small) на GPU...")
    student_model = MicroDiT_XL_2(
        input_size=64,
        caption_channels=1024,
        pos_interp_scale=1.0,
        in_channels=4
    )
    
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
    print(" Student инициализирован случайными весами")
    
    student_model.to(device, dtype=torch.float32)
    student_model.train()
    print(" Student (DiT-Small) загружен на GPU")
    
    if hasattr(student_model, 'enable_gradient_checkpointing'):
        student_model.enable_gradient_checkpointing()
        print(" Gradient checkpointing включен")
    
    print(" Создаем текстовый кодировщик...")
    text_encoder = ProperTextEncoder(device)
    
    print(" Модели загружены")
    return teacher_model, student_model, text_encoder

def consistency_distillation_step(latents, text_embeddings, teacher_model, student_model, device="cuda"):
    try:
        noise = torch.randn_like(latents)
        
        t = torch.rand(1, device=device, dtype=torch.float32)
        
        noisy_latents = latents + t * noise
        
        with torch.no_grad():
            teacher_output = teacher_model(noisy_latents.cpu(), t.cpu(), text_embeddings.cpu())
            teacher_output = teacher_output['sample'] if isinstance(teacher_output, dict) else teacher_output
            teacher_output = teacher_output.to(device)
        
        student_output = student_model(noisy_latents, t, text_embeddings)
        student_output = student_output['sample'] if isinstance(student_output, dict) else student_output
        
        loss = F.mse_loss(student_output, teacher_output)
        
        if hasattr(consistency_distillation_step, '_debug_count'):
            consistency_distillation_step._debug_count += 1
        else:
            consistency_distillation_step._debug_count = 1
            
        if consistency_distillation_step._debug_count <= 5:
            print(f" Итерация {consistency_distillation_step._debug_count}:")
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
        print(f" Ошибка в CD шаге: {e}")
        return None

def train_cd_with_pretrained_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Устройство: {device}")
    
    teacher_model, student_model, text_encoder = load_models(device)
    
    print("\n Создаем правильный датасет...")
    latents_dir = "/home/ubuntu/train/train/datadir/latents_good"
    prompts_dir = "/home/ubuntu/train/train/datadir/prompts_good"
    
    dataset = ProperDataset(latents_dir, prompts_dir, text_encoder, device)
    print(f" Датасет создан: {len(dataset)} образцов")
    
    num_epochs = 1
    max_iters = 100
    batch_size = 1
    lr = 1e-4
    
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    
    print("")
    
    print(" Включаем дополнительные оптимизации памяти...")
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("")
    
    try:
        student_model = student_model.to(memory_format=torch.channels_last)
        print(" Channels Last включен")
    except Exception as e:
        print(f"  Channels Last не поддерживается: {e}")
    
    all_losses = []
    start_time = time.time()
    
    log_memory_usage(0, 0, " СТАРТ:")
    
    print(f"\n ТЕСТОВОЕ CD ОБУЧЕНИЕ С ПРЕДОБУЧЕННЫМ TEACHER")
    print(f" Эпох: {num_epochs}, Итераций: {max_iters}")
    print(f" Общее количество итераций: {num_epochs * max_iters}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        print(f"\n ЭПОХА {epoch + 1}/{num_epochs}")
        print("=" * 50)
        
        epoch_losses = []
        
        pbar = tqdm(range(max_iters), desc=f"Эпоха {epoch + 1}/{num_epochs}")
        
        for iteration in pbar:
            try:
                sample_idx = iteration % len(dataset)
                sample = dataset[sample_idx]
                
                latents = sample['latents'].unsqueeze(0).to(device, dtype=torch.float32)
                text_embeddings = sample['text_embeddings'].to(device, dtype=torch.float32)
                prompt = sample['prompt']
                
                loss_dict = consistency_distillation_step(
                    latents, text_embeddings, teacher_model, student_model, device
                )
                
                if loss_dict is None:
                    continue
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                epoch_losses.append(loss_dict['total_loss'].item())
                all_losses.append(loss_dict['total_loss'].item())
                
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total_loss'].item():.6f}",
                    'Avg': f"{np.mean(epoch_losses):.6f}",
                    'Prompt': prompt[:20] + "..." if len(prompt) > 20 else prompt
                })
                
                if iteration % 10 == 0:
                    log_memory_usage(iteration, epoch + 1, " МОНИТОРИНГ ПАМЯТИ:")
                
            except Exception as e:
                print(f" Ошибка на итерации {iteration}: {e}")
                continue
        
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            print(f" Эпоха {epoch + 1} завершена. Средний loss: {avg_loss:.6f}")
        else:
            print(f" Эпоха {epoch + 1} завершена. Нет успешных итераций.")
    
    try:
        student_model_cpu = student_model.cpu()
        torch.save(student_model_cpu.state_dict(), 'student_test_cd_pretrained_teacher.pt')
        student_model.to(device)
        print(f" Тестовая модель сохранена: student_test_cd_pretrained_teacher.pt")
    except Exception as e:
        print(f"  Не удалось сохранить тестовую модель: {e}")
    
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(all_losses)
        plt.title('Тестовое CD обучение с предобученным Teacher - Потери')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('test_cd_pretrained_teacher_losses.png')
        print(" График сохранен: test_cd_pretrained_teacher_losses.png")
    except Exception as e:
        print(f"  Не удалось сохранить график: {e}")
    
    total_time = time.time() - start_time
    print(f"\n ТЕСТОВОЕ CD ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f" Общее время: {total_time/60:.1f} минут")
    if len(all_losses) > 0:
        print(f" Начальный loss: {all_losses[0]:.6f}")
        print(f" Финальный loss: {all_losses[-1]:.6f}")
        if all_losses[0] > 0:
            improvement = ((all_losses[0] - all_losses[-1]) / all_losses[0] * 100)
            print(f" Улучшение: {improvement:.1f}%")
        else:
            print(" Улучшение: Невозможно вычислить (начальный loss = 0)")
    else:
        print(" Нет успешных итераций обучения")
    print(f" Тестовые веса: student_test_cd_pretrained_teacher.pt")
    print(f" Тестовый график: test_cd_pretrained_teacher_losses.png")
    
    print(f"\n ТЕСТ ЗАВЕРШЕН - ФОКУС НА ОБУЧЕНИИ CD")
    print("=" * 50)

if __name__ == "__main__":
    train_cd_with_pretrained_teacher()
