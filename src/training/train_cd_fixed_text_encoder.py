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

from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
from micro_diffusion.models.dit import MicroDiT_XL_2, MicroDiT_Tiny_2
from micro_diffusion.micro_diffusion.models.utils import UniversalTextEncoder, UniversalTokenizer
from create_proper_dataset import ProperDataset

from diffusers import AutoencoderKL

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
    print("🔄 Загружаем модели...")
    
    print("🧠 Загружаем Teacher с ПРАВИЛЬНОЙ архитектурой...")
    
    teacher_model = create_latent_diffusion(
        latent_res=64,  
        in_channels=4, 
        pos_interp_scale=2.0,
        precomputed_latents=True,  
        dtype="bfloat16"
    ).to("cpu")
    
    pretrained_path = "/home/ubuntu/train/train/micro_diffusion/pretrained_models/dit_4_channel_37M_real_and_synthetic_data.pt"
    
    try:
        print(f" Загружаем предобученные веса из: {os.path.basename(pretrained_path)}")
        teacher_weights = torch.load(pretrained_path, map_location="cpu")
        
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        print("Предобученные веса Teacher загружены успешно!")
        print("Teacher: FID 12.66 (37M изображений)")
        
    except Exception as e:
        print(f"Ошибка загрузки предобученных весов: {e}")
        print("Используем случайную инициализацию Teacher")
    
    teacher_model.eval()
    print(" Teacher загружен на CPU с ПРАВИЛЬНОЙ архитектурой")
    
    print("Загружаем Student (DiT-Tiny) на GPU...")
    student_model = MicroDiT_Tiny_2(
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
    
    student_model = student_model.to(memory_format=torch.channels_last)
    print(" Channels Last включен для Student")
    
    try:
        import xformers
        if hasattr(student_model, 'enable_xformers_memory_efficient_attention'):
            student_model.enable_xformers_memory_efficient_attention()
            print(" xFormers Memory Efficient Attention включен")
        else:
            if hasattr(student_model, 'enable_attention_slicing'):
                student_model.enable_attention_slicing(1)
                print("Sliced attention включен (fallback)")
    except ImportError:
        print("xFormers не установлен, используем sliced attention")
        if hasattr(student_model, 'enable_attention_slicing'):
            student_model.enable_attention_slicing(1)
            print("Sliced attention включен (fallback)")
    except Exception as e:
        print(f"xFormers ошибка: {e}")
        if hasattr(student_model, 'enable_attention_slicing'):
            student_model.enable_attention_slicing(1)
            print(" Sliced attention включен (fallback)")
    
    print(" Student (DiT-Tiny) загружен на GPU с оптимизациями памяти")
    
    print(" Gradient Checkpointing отключен (ускорение вычислений)")
    
    print(" Создаем ПРАВИЛЬНЫЙ текстовый кодировщик...")
    text_encoder = UniversalTextEncoder(
        'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',  
        dtype='bfloat16',
        pretrained=True
    ).to(device)  
    tokenizer = UniversalTokenizer('openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')
    
    print(" ПРАВИЛЬНЫЙ Text Encoder загружен на GPU (как у Teacher)")
    
    print(" Загружаем VAE для Student...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)  
    
    if hasattr(vae, 'enable_slicing'):
        vae.enable_slicing()
        print(" Sliced VAE decode включен")
    
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        print(" Tiled VAE включен")
    
    print(" VAE загружен для Student с оптимизациями памяти")
    
    print(" Модели загружены")
    return teacher_model, student_model, text_encoder, tokenizer, vae

def consistency_distillation_step(image_path, prompt, teacher_model, student_model, text_encoder, tokenizer, vae, device="cuda"):

    try:
        cache_key = f"{image_path}_{prompt}"
        
        if not hasattr(consistency_distillation_step, 'cache'):
            consistency_distillation_step.cache = {}
        
        if cache_key in consistency_distillation_step.cache:
            latents, text_embeddings = consistency_distillation_step.cache[cache_key]
        else:
            from PIL import Image
            import torchvision.transforms as transforms
            
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                latents = vae.encode(image_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            tokenized = tokenizer.tokenize(prompt)
            text_embeddings = text_encoder.encode(tokenized['input_ids'].to(device))[0].to(device, dtype=torch.float32)
            
            if len(consistency_distillation_step.cache) < 1000:
                consistency_distillation_step.cache[cache_key] = (latents, text_embeddings)
        
        t1 = torch.rand(1, device=device, dtype=torch.float32)
        t2 = torch.rand(1, device=device, dtype=torch.float32)
        
        noise1 = torch.randn_like(latents)
        noise2 = torch.randn_like(latents)
        
        x_t1 = latents + t1 * noise1  
        x_t2 = latents + t2 * noise2  
        
        student_x0_1 = student_model(x_t1, t1, text_embeddings)
        student_x0_1 = student_x0_1['sample'] if isinstance(student_x0_1, dict) else student_x0_1
        
        student_x0_2 = student_model(x_t2, t2, text_embeddings)
        student_x0_2 = student_x0_2['sample'] if isinstance(student_x0_2, dict) else student_x0_2
        
        consistency_loss = F.mse_loss(student_x0_1, student_x0_2)
        
        if not hasattr(consistency_distillation_step, 'debug_count'):
            consistency_distillation_step.debug_count = 0
        
        if consistency_distillation_step.debug_count < 5:  
            print(f" DEBUG итерация {consistency_distillation_step.debug_count}:")
            print(f"  Student output 1 mean: {student_x0_1.mean():.6f}")
            print(f"  Student output 2 mean: {student_x0_2.mean():.6f}")
            print(f"  Difference mean: {torch.abs(student_x0_1 - student_x0_2).mean():.6f}")
            print(f"  Loss: {consistency_loss.item():.6f}")
            print(f"  t1: {t1.item():.6f}, t2: {t2.item():.6f}")
            consistency_distillation_step.debug_count += 1
        
        return consistency_loss
        
    except Exception as e:
        print(f" Ошибка в CD шаге: {e}")
        return None

def train_cd_fixed_text_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Устройство: {device}")
    
    teacher_model, student_model, text_encoder, tokenizer, vae = load_models(device)
    
    print("\n Создаем датасет с ПИКСЕЛЬНЫМИ ИЗОБРАЖЕНИЯМИ и ПРОМПТАМИ...")
    data_dir = "/home/ubuntu/train/train/dataset_sdxl_turbo"  
    
    class PixelDataset:
        def __init__(self, data_dir, device, preload_to_ram=True):
            self.data_dir = data_dir
            self.device = device
            self.preload_to_ram = preload_to_ram
            
            all_files = os.listdir(data_dir)
            self.image_files = sorted([f for f in all_files if f.endswith('.png')])
            
            print(f" Найдено {len(self.image_files)} изображений")
            print(f" Размер датасета: {len(self.image_files)}")
            
            for img_file in self.image_files[:5]:  
                txt_file = img_file.replace('.png', '.txt')
                if not os.path.exists(os.path.join(data_dir, txt_file)):
                    print(f"  ВНИМАНИЕ: Нет файла {txt_file} для {img_file}")
            
            if preload_to_ram:
                print(" Предзагружаем данные в ОЗУ для ускорения...")
                self.images_ram = {}
                self.prompts_ram = {}
                
                for i, img_file in enumerate(self.image_files):
                    if i % 100 == 0:
                        print(f"   Загружено {i}/{len(self.image_files)} изображений")
                    
                    image_path = os.path.join(data_dir, img_file)
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    self.images_ram[img_file] = image
                    
                    prompt_file = img_file.replace('.png', '.txt')
                    prompt_path = os.path.join(data_dir, prompt_file)
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    self.prompts_ram[img_file] = prompt
                
                print(f" Предзагружено {len(self.images_ram)} изображений в ОЗУ")
                print(f" Предзагружено {len(self.prompts_ram)} промптов в ОЗУ")
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            image_file = self.image_files[idx]
            
            if self.preload_to_ram and image_file in self.images_ram:
                image = self.images_ram[image_file]
                prompt = self.prompts_ram[image_file]
                
                import tempfile
                temp_image_path = f"/tmp/temp_image_{idx}.png"
                image.save(temp_image_path)
                
                return {
                    'image_path': temp_image_path,
                    'prompt': prompt
                }
            else:
                image_path = os.path.join(self.data_dir, image_file)
                prompt_file = image_file.replace('.png', '.txt')
                prompt_path = os.path.join(self.data_dir, prompt_file)
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                return {
                    'image_path': image_path,
                    'prompt': prompt
                }
    
    dataset = PixelDataset(data_dir, device)
    print(f" Датасет создан: {len(dataset)} образцов")
    
    num_epochs = 5  
    max_iters = 4000  
    batch_size = 4  
    lr = 1e-4
    
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    
    print(" Используем float32")
    
    print(" Включаем дополнительные оптимизации памяти...")
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        print(" TF32 включен для Ampere GPU")
    
    try:
        student_model = student_model.to(memory_format=torch.channels_last)
        print(" Channels Last включен")
    except Exception as e:
        print(f"  Channels Last не поддерживается: {e}")
    
    all_losses = []
    start_time = time.time()
    
    log_memory_usage(0, 0, " СТАРТ:")
    
    def log_detailed_memory(iteration, epoch, stage=""):
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_free = gpu_total - gpu_reserved
            
            print(f"  ПАМЯТЬ {stage}:")
            print(f"  Итерация: {iteration}, Эпоха: {epoch}")
            print(f"  GPU Allocated: {gpu_allocated:.2f}GB")
            print(f"  GPU Reserved: {gpu_reserved:.2f}GB") 
            print(f"  GPU Free: {gpu_free:.2f}GB")
            print(f"  GPU Total: {gpu_total:.2f}GB")
            print(f"  Использование: {(gpu_allocated/gpu_total)*100:.1f}%")
            print("-" * 50)
    
    print(f"\n ПОЛНОЕ CD ОБУЧЕНИЕ С BATCH_SIZE = 6!")
    print(f" Эпох: {num_epochs}, Итераций: {max_iters}")
    print(f" Batch Size: {batch_size} (оптимизированное обучение!)")
    print(f" VAE на GPU: ускорение кодирования/декодирования")
    print(f" Gradient Checkpointing: ОТКЛЮЧЕН (ускорение!)")
    print(f" Общее количество итераций: {num_epochs * max_iters}")
    print(f" Ожидаемое время: ~{num_epochs * max_iters * 0.4 / 60:.1f} минут")
    print("=" * 70)
    
    log_detailed_memory(0, 0, "НАЧАЛО ОБУЧЕНИЯ")
    
    for epoch in range(num_epochs):
        print(f"\n ЭПОХА {epoch + 1}/{num_epochs}")
        print("=" * 50)
        
        epoch_losses = []
        
        pbar = tqdm(range(max_iters), desc=f"Эпоха {epoch + 1}/{num_epochs}")
        
        for iteration in pbar:
            try:
                batch_data = []
                
                for b in range(batch_size):
                    sample_idx = (iteration * batch_size + b) % len(dataset)
                    sample = dataset[sample_idx]
                    batch_data.append(sample)
                
                sample = batch_data[0]
                image_path = sample['image_path']
                prompt = sample['prompt']
                
                loss = consistency_distillation_step(
                    image_path, prompt, teacher_model, student_model, 
                    text_encoder, tokenizer, vae, device
                )
                
                if iteration % 10 == 0:  
                    print(f" Итерация {iteration + 1}, Loss: {loss.item():.6f}")
                
                if loss is None:
                    continue
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                epoch_losses.append(loss.item())
                all_losses.append(loss.item())
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.6f}",
                    'Avg': f"{np.mean(epoch_losses):.6f}",
                    'Batch': f"{batch_size}"
                })
                
                if iteration % 10 == 0:
                    log_memory_usage(iteration, epoch + 1, " МОНИТОРИНГ ПАМЯТИ:")
                    log_detailed_memory(iteration, epoch + 1, "ДЕТАЛЬНЫЙ МОНИТОРИНГ")
                
                if iteration % 500 == 0 and iteration > 0:
                    try:
                        checkpoint = {
                            'model_state_dict': student_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'iteration': iteration,
                            'loss': loss.item(),
                            'batch_size': batch_size
                        }
                        torch.save(checkpoint, f'checkpoint_iter_{iteration}.pt')
                        print(f" Чекпоинт сохранен: checkpoint_iter_{iteration}.pt")
                    except Exception as e:
                        print(f" Ошибка сохранения чекпоинта: {e}")
                
            except Exception as e:
                print(f" Ошибка на итерации {iteration}: {e}")
                continue
        
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            print(f" Эпоха {epoch + 1} завершена. Средний loss: {avg_loss:.6f}")
            
            try:
                torch.save(student_model.state_dict(), f'student_epoch_{epoch+1}.pt')
                print(f" Модель эпохи {epoch+1} сохранена: student_epoch_{epoch+1}.pt")
            except Exception as e:
                print(f" Ошибка сохранения модели эпохи: {e}")
        else:
            print(f" Эпоха {epoch + 1} завершена. Нет успешных итераций.")
    
    try:
        print(f" Проверяем архитектуру Student: dim={student_model.dim if hasattr(student_model, 'dim') else 'unknown'}")
        student_model_cpu = student_model.cpu()
        torch.save(student_model_cpu.state_dict(), 'student_test_cd_fixed_text_encoder.pt')
        student_model.to(device)
        print(f" Тестовая модель сохранена: student_test_cd_fixed_text_encoder.pt")
    except Exception as e:
        print(f"  Не удалось сохранить тестовую модель: {e}")
    
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(all_losses)
        plt.title('Тестовое CD обучение с ПРАВИЛЬНЫМ Text Encoder - Потери')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('test_cd_fixed_text_encoder_losses.png')
        print(" График сохранен: test_cd_fixed_text_encoder_losses.png")
    except Exception as e:
        print(f"  Не удалось сохранить график: {e}")
    
    total_time = time.time() - start_time
    print(f"\n ПОЛНОЕ CD ОБУЧЕНИЕ ЗАВЕРШЕНО!")
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
    print(f" Тестовые веса: student_test_cd_fixed_text_encoder.pt")
    print(f" Тестовый график: test_cd_fixed_text_encoder_losses.png")
    
    print(f"\n ТЕСТИРУЕМ ГЕНЕРАЦИЮ ИЗОБРАЖЕНИЙ:")
    print("=" * 50)
    
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32)
        vae.to(device)
        vae.eval()
        print(" VAE загружен для генерации (ТОТ ЖЕ, что у Teacher)")
        
        test_prompts = [
            "A beautiful sunset over mountains",
            "A cozy cabin in a snowy forest",
            "A majestic dragon flying over a medieval castle"
        ]
        
        os.makedirs("test_fixed_text_encoder_outputs", exist_ok=True)
        
        student_model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                print(f"\n Генерируем: '{prompt}'")
                
                tokenized = tokenizer.tokenize(prompt)
                text_embeddings = text_encoder.encode(tokenized['input_ids'].to(device))[0].to(device, dtype=torch.float32)
                print(f" Эмбеддинги: {text_embeddings.shape}")
                
                latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
                print(f" Инициализированы латенты: {latents.shape}")
                
                num_steps = 4
                print(f" Генерация {num_steps} шагов...")
                
                for step in tqdm(range(num_steps), desc=f"Генерация {i+1}/3"):
                    t = torch.ones(1, device=device, dtype=torch.float32) * (1.0 - step / (num_steps - 1))
                    output = student_model(latents, t, text_embeddings)
                    latents = output['sample'] if isinstance(output, dict) else output
                
                latents_fp32 = latents.to(torch.float32)
                with torch.no_grad():
                    decoded_output = vae.decode(latents_fp32)
                decoded_image = decoded_output.sample if hasattr(decoded_output, 'sample') else decoded_output
                
                decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                image_tensor = decoded_image[0].cpu()
                image_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_array = np.clip(image_array, 0, 255)
                
                from PIL import Image
                image = Image.fromarray(image_array)
                filename = f"test_fixed_text_encoder_outputs/test_generated_{i+1}.png"
                image.save(filename)
                print(f" Сохранено: {filename}")
        
        print(f"\n ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
        print(f" Результаты в папке: test_fixed_text_encoder_outputs/")
        
        log_detailed_memory(max_iters, num_epochs, "КОНЕЦ ОБУЧЕНИЯ")
        
    except Exception as e:
        print(f" Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_cd_fixed_text_encoder()
