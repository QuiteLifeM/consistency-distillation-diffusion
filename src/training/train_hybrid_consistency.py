import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class LatentPromptDataset(Dataset):
    def __init__(self, latents_dir, prompts_dir):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        print(f"Найдено {len(self.latent_files)} латентов и {len(self.prompt_files)} промптов")
        
        if len(self.latent_files) != len(self.prompt_files):
            print(f"Несоответствие: {len(self.latent_files)} латентов vs {len(self.prompt_files)} промптов")
            min_len = min(len(self.latent_files), len(self.prompt_files))
            self.latent_files = self.latent_files[:min_len]
            self.prompt_files = self.prompt_files[:min_len]
            print(f"Используем {min_len} пар")
    
    def __len__(self):
        return len(self.latent_files)
    
    def __getitem__(self, idx):
        latent_path = os.path.join(self.latents_dir, self.latent_files[idx])
        latent = torch.load(latent_path, map_location="cpu")
        
        prompt_path = os.path.join(self.prompts_dir, self.prompt_files[idx])
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        return latent, prompt

def custom_collate(batch):
    latents, prompts = zip(*batch)
    latents = torch.stack(latents)
    return latents, list(prompts)

def get_text_embeddings(prompts, model, device="cpu"):
    tokenized = model.tokenizer.tokenize(prompts)
    input_ids = tokenized['input_ids'].to(device)
    
    with torch.no_grad():
        text_embeddings = model.text_encoder.encode(input_ids)[0]
    
    return text_embeddings

def consistency_distillation_step_hybrid(latents, text_embeddings, teacher_model, student_model):
    batch_size = latents.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
    sigma = (rnd_normal * teacher_model.edm_config.P_std + teacher_model.edm_config.P_mean).exp()
    
    latents_cpu = latents.cpu()
    text_embeddings_cpu = text_embeddings.cpu()
    sigma_cpu = sigma.cpu()
    
    noise_cpu = torch.randn_like(latents_cpu) * sigma_cpu
    noisy_latents_cpu = latents_cpu + noise_cpu
    with torch.no_grad():
        teacher_output = teacher_model.model_forward_wrapper(
            noisy_latents_cpu.float(),
            sigma_cpu,
            text_embeddings_cpu.float(),
            teacher_model.dit,
            mask_ratio=0.0
        )
        teacher_denoised = teacher_output['sample']
    
    noisy_latents_gpu = noisy_latents_cpu.to(device)
    text_embeddings_gpu = text_embeddings_cpu.to(device)
    sigma_gpu = sigma_cpu.to(device)
    teacher_denoised_gpu = teacher_denoised.to(device)
    student_output = student_model.model_forward_wrapper(
        noisy_latents_gpu.float(),
        sigma_gpu,
        text_embeddings_gpu.float(),
        student_model.dit,
        mask_ratio=0.0
    )
    student_denoised = student_output['sample']
    
    loss = nn.MSELoss()(student_denoised, teacher_denoised_gpu)
    
    return loss

def train_consistency_distillation_hybrid(dataloader, teacher_model, student_model, optimizer, num_epochs=5):
    losses_history = []
    
    print(f"\n{'='*60}")
    print(f"ГИБРИДНЫЙ Consistency Distillation")
    print(f"Teacher: CPU (MicroDiT_XL_2, 1.16B параметров)")
    print(f"Student: GPU (RTX 3090)")
    print(f"Экономия памяти: ~12GB VRAM")
    print(f"Примерное время: ~{len(dataloader) * 0.8 * num_epochs / 60:.1f} минут")
    print(f"{'='*60}\n")

    epoch_pbar = tqdm(range(num_epochs), desc="Гибридное обучение", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} эпох [{elapsed}<{remaining}]')

    for epoch in epoch_pbar:
        torch.cuda.empty_cache()
        
        epoch_loss = 0.0
        student_model.train()
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{num_epochs}", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, (latents, prompts) in enumerate(pbar):
            try:
                latents = latents.float().cuda()
                
                text_embeddings = get_text_embeddings(prompts, teacher_model, device="cuda")
                
                loss = consistency_distillation_step_hybrid(
                    latents, text_embeddings, teacher_model, student_model
                )
                
                if torch.isnan(loss):
                    print(f"\nNaN loss в батче {i}")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(student_model.dit.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                torch.cuda.empty_cache()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.6f}",
                    'Avg': f"{epoch_loss/(num_batches+1):.6f}" if num_batches > 0 else "0.000000"
                })
                    
            except Exception as e:
                print(f"\n Ошибка в батче {i}: {e}")
                continue
        
        pbar.close()
        
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            losses_history.append(avg_epoch_loss)
            
            epoch_pbar.set_postfix({
                'Loss': f"{avg_epoch_loss:.6f}",
                'Batches': f"{num_batches}/{len(dataloader)}"
            })
            
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"student_hybrid_checkpoint_epoch_{epoch+1}.pt"
                torch.save(student_model.dit.state_dict(), checkpoint_path)
                print(f"\nСохранен чекпоинт: {checkpoint_path}")
        else:
            print(f"\nЭпоха {epoch+1}: Нет успешных батчей!")
    
    epoch_pbar.close()
    
    print("Гибридное обучение завершено!")
    return losses_history

if __name__ == "__main__":
    print("Запуск ГИБРИДНОГО Consistency Distillation")
    print("Teacher: CPU |  Student: GPU")
    print("")
    
    if not torch.cuda.is_available():
        print("CUDA недоступна!")
        exit(1)
    
    print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    latents_dir = os.path.join("datadir", "latents_good")
    prompts_dir = os.path.join("datadir", "prompts_good")
    
    if not os.path.exists(latents_dir) or not os.path.exists(prompts_dir):
        print(f"Данные не найдены: {latents_dir} или {prompts_dir}")
        exit(1)
    
    print(f"Данные найдены: {latents_dir}, {prompts_dir}")
    
    print("\nСоздание модели учителя на CPU...")
    teacher_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cpu")  
    
    print("Загружаем веса учителя на CPU...")
    try:
        teacher_weights = torch.load("./micro_diffusion/micro_diffusion/trained_models/teacher.pt", map_location="cpu")
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        teacher_model.eval()
        print("Модель учителя загружена на CPU")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        exit(1)
    
    print("\nСоздание модели студента на GPU...")
    student_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cuda")  
    
    student_model.dit.load_state_dict(teacher_weights, strict=False)
    student_model.train()
    
    print("Модель студента создана и инициализирована весами учителя на GPU")
    
    optimizer = optim.Adam(student_model.dit.parameters(), lr=1e-5)
    print(f"Оптимизатор настроен с lr={1e-5}\n")
    
    print("Загрузка данных...")
    dataset = LatentPromptDataset(latents_dir, prompts_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate
    )
    print(f"DataLoader создан: {len(dataset)} сэмплов, batch_size=2, num_workers=0\n")
    
    num_epochs = 5
    print(f"Начинаем гибридное обучение на {num_epochs} эпох...")
    print(f"Примерное время: ~{len(dataloader) * 0.8 * num_epochs / 60:.1f} минут")
    
    losses = train_consistency_distillation_hybrid(
        dataloader, teacher_model, student_model, optimizer, num_epochs
    )
    
    print("\nСохранение финальной модели...")
    torch.save(student_model.dit.state_dict(), "student_final_hybrid.pt")
    print("Финальная модель сохранена: student_final_hybrid.pt")
    
    print("\nСоздание графика loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='Consistency Distillation Loss')
    plt.title('Гибридное обучение: Teacher (CPU) + Student (GPU)', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hybrid_training_loss.png', dpi=300, bbox_inches='tight')
    print("График сохранен: hybrid_training_loss.png")
    
    print(f"\nГИБРИДНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Финальный loss: {losses[-1]:.6f}")
    print(f"Модель: student_final_hybrid.pt")
    print(f"График: hybrid_training_loss.png")











