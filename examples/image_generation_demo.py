
import torch
import os
import sys
sys.path.append('/home/ubuntu/train/train')

from micro_diffusion.models.model import create_latent_diffusion
from micro_diffusion.models.model import UniversalTextEncoder
from diffusers import AutoencoderKL
import torchvision.transforms as T

def test_generation():
    device = torch.device('cuda')
    print(' ТЕСТИРУЕМ ГЕНЕРАЦИЮ ИЗОБРАЖЕНИЙ:')
    print('=' * 50)

    print(' Загружаем обученную модель...')
    student_state_dict = torch.load('student_test_cd_fixed_text_encoder.pt', map_location=device)
    
    from micro_diffusion.models.dit import DiT
    student_model = DiT(
        dim=384,
        depth=12,
        head_dim=64,
        patch_mixer_dim=256,
        patch_mixer_depth=4,
        caption_channels=4096,
    )
    student_model.load_state_dict(student_state_dict)
    student_model.to(device, dtype=torch.float32)
    student_model.eval()

    print(' Загружаем VAE...')
    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae', torch_dtype=torch.float32)
    vae.to(device)
    vae.eval()

    print(' Загружаем Text Encoder...')
    universal_text_encoder = UniversalTextEncoder(
        'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
        dtype='bfloat16', 
        pretrained=True
    )

    test_prompts = [
        'A beautiful sunset over mountains',
        'A cozy cabin in a snowy forest', 
        'A majestic dragon flying over a medieval castle'
    ]

    os.makedirs('test_fixed_text_encoder_outputs', exist_ok=True)

    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f'\n Генерируем: {prompt}')
            
            text_embeddings = universal_text_encoder.encode_text(prompt).to(torch.float32)
            
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
            
            print(f' Генерация 4 шагов...')
            for step in range(4):
                t = torch.ones(1, device=device, dtype=torch.float32) * (1.0 - step / 3.0)
                output = student_model(latents, t, text_embeddings)
                latents = output['sample'] if isinstance(output, dict) else output
                print(f' Шаг {step + 1}/4: t={t.item():.3f}')
            
            latents_fp32 = latents.to(torch.float32)
            decoded_output = vae.decode(latents_fp32)
            decoded_image = decoded_output.sample if hasattr(decoded_output, 'sample') else decoded_output
            
            image = T.ToPILImage()(decoded_image[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
            
            filename = f'test_fixed_text_encoder_outputs/test_generated_{i+1}.png'
            image.save(filename)
            print(f' Сохранено: {filename}')

    print('\n ГЕНЕРАЦИЯ ЗАВЕРШЕНА!')
    print(' Результаты в папке: test_fixed_text_encoder_outputs/')

if __name__ == "__main__":
    test_generation()
