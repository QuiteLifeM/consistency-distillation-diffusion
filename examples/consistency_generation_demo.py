
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/home/ubuntu/train/train/micro_diffusion')

def load_trained_model(device="cuda"):
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
    
    model_path = "student_final_5epochs_consistency.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        student_model.load_state_dict(checkpoint)
        student_model.to(device)
        student_model.eval()
        print(f" Модель загружена: {model_path}")
    else:
        print(f" Модель не найдена: {model_path}")
        return None
    
    return student_model

def load_vae_and_text_encoder(device="cuda"):
    vae = None
    text_encoder = None
    
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
    
    return vae, text_encoder

def generate_with_consistency_distillation(student_model, text_prompt, vae=None, text_encoder=None, device="cuda", num_steps=1):
    print(f" Генерируем изображение: '{text_prompt}'")
    print(f" Шагов: {num_steps}")
    
    if text_encoder is not None:
        with torch.no_grad():
            tokenized = text_encoder.tokenizer.tokenize([text_prompt])
            input_ids = tokenized['input_ids'].to(device)
            text_embeddings = text_encoder.encode(input_ids)[0]
            if text_embeddings.dim() == 4:
                text_embeddings = text_embeddings.squeeze(1)
    else:
        text_embeddings = torch.randn(1, 77, 1024, device=device)
        print("  Используем случайные text embeddings")
    
    latents = torch.randn(1, 4, 64, 64, device=device)
    
    print(f" Начальные латенты: {latents.shape}")
    print(f" Text embeddings: {text_embeddings.shape}")
    
    with torch.no_grad():
        if num_steps == 1:
            t = torch.ones(1, device=device)
            student_output = student_model(latents, t, text_embeddings)
            generated_latents = student_output['sample'] if isinstance(student_output, dict) else student_output
            print(" Один шаг генерации!")
            
        else:
            for step in range(num_steps):
                t = torch.ones(1, device=device) * (1.0 - step / num_steps)
                student_output = student_model(latents, t, text_embeddings)
                generated_latents = student_output['sample'] if isinstance(student_output, dict) else student_output
                
                if step < num_steps - 1:
                    latents = generated_latents
                
                print(f" Шаг {step + 1}/{num_steps}: t={t.item():.3f}")
    
    print(f" Сгенерированные латенты: {generated_latents.shape}")
    
    if vae is not None:
        with torch.no_grad():
            generated_image = vae.decode(generated_latents)
            print(f" Декодированное изображение: {generated_image.shape}")
            return generated_image
    else:
        print("  VAE не найден, возвращаем латенты")
        return generated_latents

def generate_ultra_quality(student_model, text_prompt, vae=None, text_encoder=None, device="cuda"):
    print(f" ГЕНЕРАЦИЯ МАКСИМАЛЬНОГО КАЧЕСТВА: '{text_prompt}'")
    print(f" Шагов: 64 (максимальное качество)")
    
    if text_encoder is not None:
        with torch.no_grad():
            tokenized = text_encoder.tokenizer.tokenize([text_prompt])
            input_ids = tokenized['input_ids'].to(device)
            text_embeddings = text_encoder.encode(input_ids)[0]
            if text_embeddings.dim() == 4:
                text_embeddings = text_embeddings.squeeze(1)
    else:
        text_embeddings = torch.randn(1, 77, 1024, device=device)
        print("  Используем случайные text embeddings")
    
    latents = torch.randn(1, 4, 64, 64, device=device)
    
    print(f" Начальные латенты: {latents.shape}")
    
    with torch.no_grad():
        for step in range(64):
            t = torch.ones(1, device=device) * (1.0 - step / 63.0)
            
            student_output = student_model(latents, t, text_embeddings)
            generated_latents = student_output['sample'] if isinstance(student_output, dict) else student_output
            
            latents = generated_latents
            
            if step % 8 == 0:
                print(f" Шаг {step + 1}/64: t={t.item():.3f}")
    
    print(f" Финальные латенты: {generated_latents.shape}")
    
    if vae is not None:
        with torch.no_grad():
            generated_image = vae.decode(generated_latents)
            print(f" Декодированное изображение: {generated_image.shape}")
            return generated_image
    else:
        return generated_latents

def test_consistency_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Устройство: {device}")
    
    student_model = load_trained_model(device)
    if student_model is None:
        return
    
    vae, text_encoder = load_vae_and_text_encoder(device)
    
    test_prompts = [
        "A beautiful sunset over mountains",
        "A cute cat playing with yarn", 
        "A futuristic city at night",
        "A majestic eagle soaring through clouds",
        "A cozy cabin in a snowy forest"
    ]
    
    print(f"\n ТЕСТИРУЕМ ГЕНЕРАЦИЮ")
    print("=" * 50)
    
    output_dir = "consistency_generation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    num_steps_list = [1, 2, 4, 8, 16, 32]
    
    for num_steps in num_steps_list:
        print(f"\n ТЕСТ: {num_steps} шаг(ов) генерации")
        print("-" * 30)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n Промпт {i+1}: {prompt}")
            
            try:
                start_time = time.time()
                generated = generate_with_consistency_distillation(
                    student_model, prompt, vae, text_encoder, device, num_steps
                )
                generation_time = time.time() - start_time
                
                print(f" Время генерации: {generation_time:.2f} секунд")
                
                if isinstance(generated, torch.Tensor) and generated.dim() == 4:
                    image_tensor = generated[0].cpu()
                    image_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_array)
                    
                    filename = f"{output_dir}/test_{i+1}_{num_steps}steps.png"
                    image.save(filename)
                    print(f" Сохранено: {filename}")
                    
                else:
                    filename = f"{output_dir}/test_{i+1}_{num_steps}steps_latents.pt"
                    torch.save(generated, filename)
                    print(f" Сохранены латенты: {filename}")
                
            except Exception as e:
                print(f" Ошибка при генерации: {e}")
                continue
    
    print(f"\n ТЕСТ МАКСИМАЛЬНОГО КАЧЕСТВА (64 шага)")
    print("=" * 50)
    
    ultra_prompts = [
        "A majestic dragon flying over a medieval castle",
        "A cyberpunk cityscape with neon lights and flying cars",
        "A serene Japanese garden with cherry blossoms and a koi pond"
    ]
    
    for i, prompt in enumerate(ultra_prompts):
        print(f"\n УЛЬТРА-КАЧЕСТВО {i+1}: {prompt}")
        
        try:
            start_time = time.time()
            generated = generate_ultra_quality(
                student_model, prompt, vae, text_encoder, device
            )
            generation_time = time.time() - start_time
            
            print(f" Время генерации: {generation_time:.2f} секунд")
            
            if isinstance(generated, torch.Tensor) and generated.dim() == 4:
                image_tensor = generated[0].cpu()
                image_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image = Image.fromarray(image_array)
                
                filename = f"{output_dir}/ultra_quality_{i+1}_64steps.png"
                image.save(filename)
                print(f" УЛЬТРА-КАЧЕСТВО сохранено: {filename}")
                
            else:
                filename = f"{output_dir}/ultra_quality_{i+1}_64steps_latents.pt"
                torch.save(generated, filename)
                print(f" УЛЬТРА-КАЧЕСТВО латенты: {filename}")
                
        except Exception as e:
            print(f" Ошибка при ультра-генерации: {e}")
            continue
    
    print(f"\n ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f" Результаты в папке: {output_dir}/")
    
    print(f"\n АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 30)
    
    for num_steps in num_steps_list:
        print(f"\n {num_steps} шаг(ов):")
        for i in range(len(test_prompts)):
            filename = f"{output_dir}/test_{i+1}_{num_steps}steps.png"
            if os.path.exists(filename):
                print(f"   test_{i+1}_{num_steps}steps.png")
            else:
                print(f"   test_{i+1}_{num_steps}steps.png (не создан)")
    
    print(f"\n УЛЬТРА-КАЧЕСТВО (64 шага):")
    for i in range(len(ultra_prompts)):
        filename = f"{output_dir}/ultra_quality_{i+1}_64steps.png"
        if os.path.exists(filename):
            print(f"   ultra_quality_{i+1}_64steps.png")
        else:
            print(f"   ultra_quality_{i+1}_64steps.png (не создан)")

def main():
    print(" ТЕСТ ГЕНЕРАЦИИ CONSISTENCY DISTILLATION")
    print("=" * 60)
    print("")
    print("")
    print("")
    print("")
    print("=" * 60)
    
    try:
        test_consistency_generation()
        
        print(f"\n ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print(f" Проверьте папку: consistency_generation_outputs/")
        
    except Exception as e:
        print(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
