
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import time
from datetime import datetime

sys.path.append('./micro_diffusion')
from micro_diffusion.models.model import create_latent_diffusion

class LatentPromptDataset(Dataset):
    def __init__(self, latents_dir, prompts_dir):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        print(f"Найдено {len(self.latent_files)} латентов и {len(self.prompt_files)} промптов")
        
        if len(self.latent_files) != len(self.prompt_files):
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
    
    latents_clean = []
    for latent in latents:
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        elif latent.dim() == 3:
            pass
        latents_clean.append(latent)
    
    latents = torch.stack(latents_clean)
    return latents, list(prompts)

def get_text_embeddings(prompts, model, device="cpu"):
    tokenized = model.tokenizer.tokenize(prompts)
    input_ids = tokenized['input_ids'].to(device)
    
    with torch.no_grad():
        text_embeddings = model.text_encoder.encode(input_ids)[0]
    
    if text_embeddings.dim() == 4:
        text_embeddings = text_embeddings.squeeze(1)
    
    return text_embeddings

def euler_step(model, x, t_cur, t_next, text_embeddings):
    if x.dim() != 4:
        raise ValueError(f" euler_step: x должен быть 4D [B,C,H,W], получен {x.shape}")
    if text_embeddings.dim() != 3:
        raise ValueError(f" euler_step: text_embeddings должен быть 3D [B,seq,dim], получен {text_embeddings.shape}")
    
    if t_cur.dim() == 0:
        t_cur_batch = t_cur.unsqueeze(0)
    elif t_cur.dim() == 1:
        t_cur_batch = t_cur
    else:
        t_cur_batch = t_cur.squeeze()
    
    output = model.model_forward_wrapper(
        x.float(),
        t_cur_batch,
        text_embeddings.float(),
        model.dit,
        mask_ratio=0.0
    )
    denoised = output['sample']
    
    t_cur_val = t_cur if t_cur.dim() == 0 else t_cur.view(-1, 1, 1, 1)
    t_next_val = t_next if t_next.dim() == 0 else t_next.view(-1, 1, 1, 1)
    
    if (t_cur_val > 0).all():
        d = (x - denoised) / t_cur_val
        x_next = x + d * (t_next_val - t_cur_val)
    else:
        x_next = denoised
    
    return x_next, denoised

def true_consistency_distillation_step(latents, text_embeddings, teacher_model, student_model):
    if latents.dim() != 4:
        raise ValueError(f" true_consistency_distillation_step: latents должен быть 4D [B,C,H,W], получен {latents.shape}")
    
    batch_size = latents.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sigma_min = teacher_model.edm_config.sigma_min
    sigma_max = teacher_model.edm_config.sigma_max
    
    N = 18
    n = torch.randint(1, N, (batch_size,), device=device)
    
    t_n = sigma_min + (n / (N - 1)) * (sigma_max - sigma_min)
    t_n_minus_1 = sigma_min + ((n - 1) / (N - 1)) * (sigma_max - sigma_min)
    
    t_n = torch.clamp(t_n, sigma_min, sigma_max)
    t_n_minus_1 = torch.clamp(t_n_minus_1, sigma_min, sigma_max)
    
    t_n = t_n.view(-1, 1, 1, 1)
    t_n_minus_1 = t_n_minus_1.view(-1, 1, 1, 1)
    
    noise = torch.randn_like(latents)
    noisy_latents_tn = latents + noise * t_n
    noisy_latents_tn1 = latents + noise * t_n_minus_1
    noisy_tn_cpu = noisy_latents_tn.cpu()
    text_emb_cpu = text_embeddings.cpu()
    t_n_cpu = t_n.view(-1).cpu()
    t_n1_cpu = t_n_minus_1.view(-1).cpu()
    
    with torch.no_grad():
        teacher_stepped, _ = euler_step(
            teacher_model,
            noisy_tn_cpu,
            t_n_cpu,
            t_n1_cpu,
            text_emb_cpu
        )
    
    noisy_tn_gpu = noisy_latents_tn.to(device)
    noisy_tn1_gpu = noisy_latents_tn1.to(device)
    text_emb_gpu = text_embeddings.to(device)
    t_n_gpu = t_n.view(-1).to(device)
    t_n1_gpu = t_n_minus_1.view(-1).to(device)
    teacher_target = teacher_stepped.to(device)

    def student_forward_tn(x, t, emb):
        output = student_model.model_forward_wrapper(
            x.float(), t, emb.float(),
            student_model.dit, mask_ratio=0.0
        )
        return output['sample']
    
    student_pred_from_tn = checkpoint(
        student_forward_tn,
        noisy_tn_gpu,
        t_n_gpu,
        text_emb_gpu,
        use_reentrant=False
    )
    
    def student_forward_tn1(x, t, emb):
        output = student_model.model_forward_wrapper(
            x.float(), t, emb.float(),
            student_model.dit, mask_ratio=0.0
        )
        return output['sample']
    
    student_pred_from_tn1 = checkpoint(
        student_forward_tn1,
        noisy_tn1_gpu,
        t_n1_gpu,
        text_emb_gpu,
        use_reentrant=False
    )
    
    loss_tn = nn.MSELoss()(student_pred_from_tn, teacher_target)
    loss_tn1 = nn.MSELoss()(student_pred_from_tn1, teacher_target)
    loss_consistency = nn.MSELoss()(student_pred_from_tn, student_pred_from_tn1.detach())
    
    loss_tn = torch.clamp(loss_tn, 0, 10.0)
    loss_tn1 = torch.clamp(loss_tn1, 0, 10.0)
    loss_consistency = torch.clamp(loss_consistency, 0, 1.0)
    
    total_loss = loss_tn + loss_tn1 + 0.5 * loss_consistency
    
    return total_loss, {
        'loss_tn': loss_tn.item(),
        'loss_tn1': loss_tn1.item(),
        'loss_consistency': loss_consistency.item()
    }

def train_5_epochs_consistency_distillation(dataloader, teacher_model, student_model, optimizer, num_epochs=5):
    all_losses = []
    epoch_losses = []
    
    print(f"\n{'='*80}")
    print(f"ПОЛНОЕ обучение True Consistency Distillation")
    print(f"Цель: {num_epochs} эпох, batch_size=1")
    print(f"Сохранение чекпоинтов каждую эпоху")
    print(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    student_model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nЭПОХА {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        epoch_losses = []
        iter_count = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{num_epochs}")
        
        for latents, prompts in pbar:
            try:
                latents = latents.float().cuda()
                text_embeddings = get_text_embeddings(prompts, teacher_model, device="cpu")
                
                loss, metrics = true_consistency_distillation_step(
                    latents, text_embeddings, teacher_model, student_model
                )
                
                if torch.isnan(loss):
                    print(f"\n NaN loss в итерации {iter_count}")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.dit.parameters(), max_norm=0.5)
                optimizer.step()
                
                torch.cuda.empty_cache()
                if iter_count % 10 == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                epoch_losses.append(loss.item())
                all_losses.append(loss.item())
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'L_cons': f"{metrics['loss_consistency']:.4f}",
                    'Avg': f"{np.mean(epoch_losses[-10:]):.4f}" if len(epoch_losses) >= 10 else f"{loss.item():.4f}"
                })
                
                iter_count += 1
                
            except Exception as e:
                print(f"\n Ошибка в итерации {iter_count}: {e}")
                continue
        
        pbar.close()
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nЭпоха {epoch + 1} завершена:")
        print(f"    Средний loss: {avg_loss:.6f}")
        print(f"    Итераций: {len(epoch_losses)}")
        print(f"    Время эпохи: {(time.time() - start_time) / (epoch + 1):.1f} сек")
        
        checkpoint_path = f"student_epoch_{epoch + 1}.pt"
        torch.save(student_model.dit.state_dict(), checkpoint_path)
        print(f"    Чекпоинт сохранен: {checkpoint_path}")
        plt.figure(figsize=(12, 8))
        plt.plot(all_losses, 'b-', linewidth=1)
        plt.title(f'Consistency Distillation Loss (эпоха {epoch + 1})')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'loss_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    График сохранен: loss_epoch_{epoch + 1}.png")
    
    total_time = time.time() - start_time
    print(f"\n Общее время обучения: {total_time/60:.1f} минут")
    print(f" Среднее время на эпоху: {total_time/num_epochs/60:.1f} минут")
    
    return all_losses

def generate_image(model, prompt, num_steps=4, device="cuda", seed=42):
    model.eval()
    torch.manual_seed(seed)
    
    with torch.no_grad():
        tokenized = model.tokenizer.tokenize([prompt])
        input_ids = tokenized['input_ids'].to(device)
        text_embeddings = model.text_encoder.encode(input_ids)[0]
        
        latents = torch.randn(1, 4, 64, 64, device=device) * model.edm_config.sigma_max
        
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (model.edm_config.sigma_max ** (1 / model.edm_config.rho) + 
                   step_indices / (num_steps - 1) * 
                   (model.edm_config.sigma_min ** (1 / model.edm_config.rho) - 
                    model.edm_config.sigma_max ** (1 / model.edm_config.rho))
                  ) ** model.edm_config.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        x = latents
        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            
            output = model.model_forward_wrapper(
                x.float(), t_cur.unsqueeze(0), text_embeddings.float(),
                model.dit, mask_ratio=0.0
            )
            denoised = output['sample']
            
            d = (x - denoised) / t_cur if t_cur > 0 else torch.zeros_like(x)
            x = x + d * (t_next - t_cur)
        
        x_scaled = (x / 0.13025).to(torch.bfloat16)
        images = model.vae.decode(x_scaled).sample
        
        images = (images / 2 + 0.5).clamp(0, 1).float()
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        
        return Image.fromarray(images[0])

def main():
    print("")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA недоступна!")
        exit(1)
    
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Доступно VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    latents_dir = os.path.join("datadir", "latents_good")
    prompts_dir = os.path.join("datadir", "prompts_good")
    
    if not os.path.exists(latents_dir) or not os.path.exists(prompts_dir):
        print(f"Данные не найдены")
        exit(1)
    
    print("\nЗагрузка Teacher...")
    teacher_model = create_latent_diffusion(
        latent_res=64, in_channels=4, pos_interp_scale=2.0,
        precomputed_latents=False, dtype="bfloat16"
    ).to("cpu")
    
    teacher_weights = torch.load("./micro_diffusion/micro_diffusion/trained_models/teacher.pt", map_location="cpu")
    teacher_model.dit.load_state_dict(teacher_weights, strict=False)
    teacher_model.eval()
    print("Teacher загружен")
    
    print("\nСоздание Student...")
    student_model = create_latent_diffusion(
        latent_res=64, in_channels=4, pos_interp_scale=2.0,
        precomputed_latents=False, dtype="bfloat16"
    ).to("cuda")
    
    student_model.dit.load_state_dict(teacher_weights, strict=False)
    student_model.train()
    
    if hasattr(student_model.dit, 'enable_gradient_checkpointing'):
        student_model.dit.enable_gradient_checkpointing()
    
    print("Student создан")
    
    optimizer = optim.SGD(student_model.dit.parameters(), lr=1e-4, momentum=0.9)
    
    dataset = LatentPromptDataset(latents_dir, prompts_dir)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate
    )
    
    print("")
    losses = train_5_epochs_consistency_distillation(dataloader, teacher_model, student_model, optimizer, num_epochs=5)
    
    final_checkpoint_path = "student_final_5epochs.pt"
    torch.save(student_model.dit.state_dict(), final_checkpoint_path)
    print(f"\nФинальные веса сохранены: {final_checkpoint_path}")
    plt.figure(figsize=(15, 10))
    plt.plot(losses, 'b-', linewidth=1)
    plt.title('Полное обучение True Consistency Distillation (5 эпох)', fontsize=16)
    plt.xlabel('Итерация', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('final_loss_5epochs.png', dpi=300, bbox_inches='tight')
    print("")
    
    print("\nГенерация тестовых изображений...")
    test_prompts = [
        "A beautiful sunset over mountains",
        "A cute cat playing with yarn",
        "A futuristic city at night",
        "A majestic eagle soaring through clouds",
        "A cozy cabin in a snowy forest"
    ]
    
    os.makedirs("final_5epochs_outputs", exist_ok=True)
    
    for i, prompt in enumerate(test_prompts):
        print(f"  {i+1}. '{prompt}'")
        image = generate_image(student_model, prompt, num_steps=4, device="cuda")
        output_path = f"final_5epochs_outputs/test_{i+1}.png"
        image.save(output_path)
        print(f"      {output_path}")
    
    print("\n" + "=" * 80)
    print("")
    print(f"Начальный loss: {losses[0]:.6f}")
    print(f"Финальный loss: {losses[-1]:.6f}")
    print(f"Изменение: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    print(f"Финальные веса: {final_checkpoint_path}")
    print(f"Финальный график: final_loss_5epochs.png")
    print(f"Тестовые изображения: final_5epochs_outputs/")
    print("=" * 80)

if __name__ == "__main__":
    main()




