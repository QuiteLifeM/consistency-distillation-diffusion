import os
import torch
import glob

def check_latent_files(latents_dir):
    """Проверяем файлы латентов на наличие NaN"""
    print(f"Проверяем латенты в {latents_dir}")
    
    # Получаем список файлов
    latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
    print(f"Найдено {len(latent_files)} файлов латентов")
    
    # Проверяем первые 10 файлов
    nan_files = []
    good_files = []
    
    for i, filename in enumerate(latent_files[:10]):
        filepath = os.path.join(latents_dir, filename)
        try:
            latent = torch.load(filepath)
            print(f"{filename}: shape={latent.shape}, dtype={latent.dtype}")
            print(f"   min={latent.min():.6f}, max={latent.max():.6f}")
            
            if torch.isnan(latent).any():
                print(f"   СОДЕРЖИТ NaN!")
                nan_files.append(filename)
            else:
                print(f"   OK")
                good_files.append(filename)
                
        except Exception as e:
            print(f"   Ошибка загрузки: {e}")
            nan_files.append(filename)
    
    print(f"\nРезультаты:")
    print(f"Хороших файлов: {len(good_files)}")
    print(f"Плохих файлов: {len(nan_files)}")
    
    if nan_files:
        print(f"Файлы с NaN: {nan_files}")
    
    return good_files, nan_files

if __name__ == "__main__":
    latents_dir = r"C:\newTry2\train\datadir\latents_good"
    good_files, nan_files = check_latent_files(latents_dir)
