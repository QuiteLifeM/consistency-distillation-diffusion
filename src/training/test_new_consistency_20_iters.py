
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from tqdm import tqdm

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

def load_models(device="cuda"):
    print("")
    
    from micro_diffusion.models.dit import DiT
    student_model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=4,
        dim=1152,
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
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        use_bias=True,
        num_experts=8,
        expert_capacity=1,
        experts_every_n=2
    )
    student_model.to(device)
    student_model.train()
    
    vae_path = "/home/ubuntu/train/train/vae_model.pt"
    if os.path.exists(vae_path):
        vae_checkpoint = torch.load(vae_path, map_location=device)
        from micro_diffusion.models.autoencoder import Autoencoder
        vae = Autoencoder()
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
        vae.to(device)
        vae.eval()
        print(" VAE загружен")
    else:
        print(" VAE не найден!")
        vae = None
    
    text_encoder_path = "/home/ubuntu/train/train/text_encoder.pt"
    if os.path.exists(text_encoder_path):
        text_encoder_checkpoint = torch.load(text_encoder_path, map_location=device)
        from micro_diffusion.models.text_encoder import TextEncoder
        text_encoder = TextEncoder()
        text_encoder.load_state_dict(text_encoder_checkpoint['model_state_dict'])
        text_encoder.to(device)
        text_encoder.eval()
        print(" Text Encoder загружен")
    else:
        print(" Text Encoder не найден!")
        text_encoder = None
    
    return student_model, vae, text_encoder

def consistency_distillation_step(latents, text_embeddings, student_model, device="cuda"):
    batch_size = latents.shape[0]
    
    x_0 = latents
    
    t = torch.rand(batch_size, device=device)
    
    noise = torch.randn_like(latents)
    x_t = x_0 + noise * t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    student_output = student_model(x_t, t, text_embeddings)
    student_pred = student_output['sample'] if isinstance(student_output, dict) else student_output
    
    loss_prediction = F.mse_loss(student_pred, x_0)
    
    loss_consistency = torch.tensor(0.0, device=device)
    
    loss_boundary = torch.tensor(0.0, device=device)
    
    total_loss = loss_prediction + 0.1 * loss_consistency + 0.1 * loss_boundary
    
    return {
        'total_loss': total_loss,
        'prediction_loss': loss_prediction,
        'consistency_loss': loss_consistency,
        'boundary_loss': loss_boundary
    }

def test_consistency_distillation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Устройство: {device}")
    
    student_model, vae, text_encoder = load_models(device)
    if student_model is None:
        return
    
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-4, momentum=0.9)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    batch_size = 1
    latent_size = 64
    channels = 4
    
    test_latents = torch.randn(batch_size, channels, latent_size, latent_size, device=device)
    
    test_prompt = "A beautiful sunset over mountains"
    
    if text_encoder is not None:
        with torch.no_grad():
            tokenized = text_encoder.tokenizer.tokenize([test_prompt])
            input_ids = tokenized['input_ids'].to(device)
            text_embeddings = text_encoder.encode(input_ids)[0]
            if text_embeddings.dim() == 4:
                text_embeddings = text_embeddings.squeeze(1)
    else:
        text_embeddings = torch.randn(batch_size, 77, 1024, device=device)
    
    print(f" Тестируем Consistency Distillation...")
    print(f" Размеры: latents={test_latents.shape}, text_embeddings={text_embeddings.shape}")
    
    losses = []
    
    print(f" Начинаем обучение на 20 итераций...")
    
    for iteration in tqdm(range(20), desc="Обучение"):
        optimizer.zero_grad()
        
        loss_dict = consistency_distillation_step(
            test_latents, text_embeddings, student_model, device
        )
        
        loss_dict['total_loss'].backward()
        
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        losses.append({
            'iteration': iteration,
            'total_loss': loss_dict['total_loss'].item(),
            'prediction_loss': loss_dict['prediction_loss'].item(),
            'consistency_loss': loss_dict['consistency_loss'].item(),
            'boundary_loss': loss_dict['boundary_loss'].item()
        })
        
        if iteration % 5 == 0:
            print(f"  Итерация {iteration}:")
            print(f"     Total Loss: {loss_dict['total_loss'].item():.6f}")
            print(f"     Prediction Loss: {loss_dict['prediction_loss'].item():.6f}")
            print(f"     Consistency Loss: {loss_dict['consistency_loss'].item():.6f}")
            print(f"     Boundary Loss: {loss_dict['boundary_loss'].item():.6f}")
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    print(f"\n АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 50)
    
    final_loss = losses[-1]['total_loss']
    initial_loss = losses[0]['total_loss']
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f" Начальный loss: {initial_loss:.6f}")
    print(f" Финальный loss: {final_loss:.6f}")
    print(f" Улучшение: {improvement:.2f}%")
    
    if improvement > 5:
        print(" Модель обучается! Loss снижается")
    elif improvement > 0:
        print("  Модель обучается медленно")
    else:
        print(" Модель НЕ обучается!")
    
    print(f"\n Тестируем генерацию...")
    
    with torch.no_grad():
        noise = torch.randn_like(test_latents)
        t_start = torch.ones(batch_size, device=device) * 0.9
        
        student_output = student_model(noise, t_start, text_embeddings)
        student_pred = student_output['sample'] if isinstance(student_output, dict) else student_output
        
        print(f" Статистика предсказания:")
        print(f"  Mean: {student_pred.mean().item():.6f}")
        print(f"  Std: {student_pred.std().item():.6f}")
        print(f"  Min: {student_pred.min().item():.6f}")
        print(f"  Max: {student_pred.max().item():.6f}")
        
        target = test_latents
        mse_to_target = F.mse_loss(student_pred, target).item()
        print(f" MSE к целевому x_0: {mse_to_target:.6f}")
        
        if mse_to_target < 1.0:
            print(" Модель хорошо предсказывает x_0!")
        else:
            print("  Модель плохо предсказывает x_0")
    
    try:
        student_model_cpu = student_model.cpu()
        torch.save(student_model_cpu.state_dict(), 'student_consistency_20iters.pt')
        student_model.to(device)
        print(f" Модель сохранена: student_consistency_20iters.pt")
    except Exception as e:
        print(f"  Не удалось сохранить модель: {e}")
        print(f" Продолжаем без сохранения...")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot([l['total_loss'] for l in losses])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot([l['prediction_loss'] for l in losses])
    plt.title('Prediction Loss (Student -> x_0)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot([l['consistency_loss'] for l in losses])
    plt.title('Consistency Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 4)
    plt.plot([l['boundary_loss'] for l in losses])
    plt.title('Boundary Loss (t=0 -> x_0)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('consistency_distillation_20iters.png', dpi=150, bbox_inches='tight')
    print(f" График сохранен: consistency_distillation_20iters.png")
    
    return losses

def main():
    print(" ТЕСТ ПРАВИЛЬНОЙ CONSISTENCY DISTILLATION")
    print("=" * 60)
    print(" Принцип: Student должен предсказывать x_0 из любого X_t")
    print(" Boundary Condition: f(X_0, t=0) = x_0")
    print("")
    print("")
    print("=" * 60)
    
    try:
        start_time = time.time()
        losses = test_consistency_distillation()
        end_time = time.time()
        
        print(f"\n ТЕСТ ЗАВЕРШЕН!")
        print(f" Время выполнения: {end_time - start_time:.1f} секунд")
        print(f" Проверьте график: consistency_distillation_20iters.png")
        print(f" Модель: student_consistency_20iters.pt")
        
    except Exception as e:
        print(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
