#!/usr/bin/env python3

import torch
import os
import sys
sys.path.append('/home/ubuntu/train/train')

from micro_diffusion.models.model import DiT
from micro_diffusion.models.model import UniversalTextEncoder
from diffusers import AutoencoderKL
import torchvision.transforms as T

def test_teacher_simple():
    device = torch.device('cuda')
    print('🎨 ПРОСТОЙ ТЕСТ TEACHER:')
    print('=' * 50)

    # Загружаем VAE
    print('🔄 Загружаем VAE...')
    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae', torch_dtype=torch.float32)
    vae.to(device)
    vae.eval()

    # Загружаем Text Encoder
    print('🔄 Загружаем Text Encoder...')
    text_encoder = UniversalTextEncoder(
        'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
        dtype='bfloat16', 
        pretrained=True
    )

    # Загружаем Teacher DiT веса
    print('🔄 Загружаем Teacher DiT веса...')
    teacher_weights = torch.load('/home/ubuntu/train/train/dit_4_channel_37M_real_and_synthetic_data.pt', map_location='cpu')
    print(f"📊 Загружено весов: {len(teacher_weights)} ключей")
    
    # Создаем DiT модель (попробуем разные параметры)
    print('🔄 Создаем DiT модель...')
    try:
        # Попробуем стандартные параметры DiT-XL
        teacher_dit = DiT(
            dim=1152,  # DiT-XL размер
            depth=28,  # DiT-XL глубина
            patch_size=2,
            num_classes=0,  # Без классов
            learn_sigma=False,
            cond_dropout=0.0,
            num_heads=16,
            head_dim=72,
            mlp_ratio=4.0,
            qkv_bias=True,
            patch_mixer_dim=256,
            patch_mixer_depth=4,
            caption_channels=4096
        )
        print("✅ DiT модель создана")
    except Exception as e:
        print(f"❌ Ошибка создания DiT: {e}")
        return

    # Загружаем веса
    print('🔄 Загружаем веса в DiT...')
    try:
        teacher_dit.load_state_dict(teacher_weights, strict=False)
        teacher_dit.to(device)
        teacher_dit.eval()
        print("✅ Веса загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки весов: {e}")
        return

    # Тестовые промпты
    test_prompts = [
        'A beautiful sunset over mountains',
        'A cozy cabin in a snowy forest', 
        'A majestic dragon flying over a medieval castle'
    ]

    os.makedirs('test_teacher_simple_outputs', exist_ok=True)

    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f'\n📝 Teacher генерирует: {prompt}')
            
            # Получаем текстовые эмбеддинги
            text_embeddings = text_encoder(prompt).to(torch.float32)
            print(f"📊 Text embeddings shape: {text_embeddings.shape}")
            
            # Инициализируем латенты (случайные)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
            print(f"📊 Latents shape: {latents.shape}")
            
            # Генерируем (20 шагов для Teacher)
            print(f'🎨 Teacher генерация 20 шагов...')
            for step in range(20):
                t = torch.ones(1, device=device, dtype=torch.float32) * (1.0 - step / 19.0)
                
                # Teacher inference
                with torch.no_grad():
                    # Прямой вызов DiT
                    teacher_output = teacher_dit(
                        latents, t, text_embeddings
                    )
                    
                    # Обновляем латенты
                    latents = teacher_output['sample'] if isinstance(teacher_output, dict) else teacher_output
                
                if step % 5 == 0:
                    print(f'🔄 Шаг {step + 1}/20: t={t.item():.3f}, output_mean={latents.mean().item():.4f}')
            
            # Декодируем в изображение
            latents_fp32 = latents.to(torch.float32)
            decoded_output = vae.decode(latents_fp32)
            decoded_image = decoded_output.sample if hasattr(decoded_output, 'sample') else decoded_output
            
            # Конвертируем в PIL
            image = T.ToPILImage()(decoded_image[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
            
            # Сохраняем
            filename = f'test_teacher_simple_outputs/teacher_simple_{i+1}.png'
            image.save(filename)
            print(f'💾 Сохранено: {filename}')

    print('\n🎨 TEACHER ПРОСТОЙ ТЕСТ ЗАВЕРШЕН!')
    print('📁 Результаты в папке: test_teacher_simple_outputs/')

if __name__ == "__main__":
    test_teacher_simple()