
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

def simple_consistency_test():
    print(" ПРОСТОЙ ТЕСТ CONSISTENCY DISTILLATION")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Устройство: {device}")
    
    from micro_diffusion.models.dit import DiT
    
    student_model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        cond_drop_prob=0.1,
    )
    student_model.to(device)
    student_model.train()
    
    print(" Student модель создана")
    
    batch_size = 1
    latent_size = 64
    channels = 4
    
    x_0 = torch.randn(batch_size, channels, latent_size, latent_size, device=device)
    
    text_embeddings = torch.randn(batch_size, 77, 1152, device=device)
    
    print(f" Размеры: x_0={x_0.shape}, text_embeddings={text_embeddings.shape}")
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    print(" Начинаем тестирование...")
    
    losses = []
    
    for iteration in range(10):
        optimizer.zero_grad()
        
        t = torch.rand(batch_size, device=device)
        
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        student_pred = student_model(x_t, t, text_embeddings)
        
        loss = F.mse_loss(student_pred, x_0)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if iteration % 2 == 0:
            print(f"  Итерация {iteration}: Loss = {loss.item():.6f}")
    
    print(f"\n РЕЗУЛЬТАТЫ:")
    print(f"  Начальный loss: {losses[0]:.6f}")
    print(f"  Финальный loss: {losses[-1]:.6f}")
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Улучшение: {improvement:.2f}%")
    
    if improvement > 5:
        print(" Модель обучается! Loss снижается")
    elif improvement > 0:
        print("  Модель обучается медленно")
    else:
        print(" Модель НЕ обучается!")
    
    print(f"\n Тестируем предсказание...")
    
    with torch.no_grad():
        noise = torch.randn_like(x_0)
        t_test = torch.ones(batch_size, device=device) * 0.5
        
        student_pred = student_model(noise, t_test, text_embeddings)
        
        mse_to_target = F.mse_loss(student_pred, x_0).item()
        print(f" MSE к целевому x_0: {mse_to_target:.6f}")
        
        if mse_to_target < 1.0:
            print(" Модель хорошо предсказывает x_0!")
        else:
            print("  Модель плохо предсказывает x_0")
    
    print(f"\n ТЕСТ ЗАВЕРШЕН!")
    return losses

if __name__ == "__main__":
    try:
        losses = simple_consistency_test()
    except Exception as e:
        print(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()

    