"""
Проверка окружения для Consistency Distillation
"""
import sys
import os

print("="*60)
print("ПРОВЕРКА ОКРУЖЕНИЯ")
print("="*60)

# 1. Проверка Python
print(f"Python версия: {sys.version}")
print(f"Python путь: {sys.executable}")

# 2. Проверка PyTorch
try:
    import torch
    print(f"\n✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA версия: {torch.version.cuda}")
        print(f"✅ GPU устройств: {torch.cuda.device_count()}")
        print(f"✅ Текущее устройство: {torch.cuda.current_device()}")
        print(f"✅ Имя GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError as e:
    print(f"❌ PyTorch не найден: {e}")

# 3. Проверка micro_diffusion
try:
    from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
    print(f"\n✅ micro_diffusion импортирован успешно")
except ImportError as e:
    print(f"❌ micro_diffusion не найден: {e}")

# 4. Проверка composer
try:
    import composer
    print(f"✅ composer импортирован успешно")
except ImportError as e:
    print(f"❌ composer не найден: {e}")

# 5. Проверка данных
latents_dir = r"C:\newTry2\train\datadir\latents"
prompts_dir = r"C:\newTry2\train\datadir\prompts"

print(f"\n📁 Проверка данных:")
print(f"   Латенты: {os.path.exists(latents_dir)} ({len(os.listdir(latents_dir)) if os.path.exists(latents_dir) else 0} файлов)")
print(f"   Промпты: {os.path.exists(prompts_dir)} ({len(os.listdir(prompts_dir)) if os.path.exists(prompts_dir) else 0} файлов)")

# 6. Проверка весов учителя
teacher_path = "./micro_diffusion/trained_models/teacher.pt"
print(f"\n🤖 Проверка весов учителя:")
print(f"   Файл существует: {os.path.exists(teacher_path)}")
if os.path.exists(teacher_path):
    size_mb = os.path.getsize(teacher_path) / 1024**2
    print(f"   Размер файла: {size_mb:.1f} MB")

print("\n" + "="*60)
print("РЕКОМЕНДАЦИИ:")
print("="*60)

if not torch.cuda.is_available():
    print("❌ CUDA недоступна - нужна GPU версия PyTorch")
    print("   Решение: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
else:
    print("✅ CUDA работает - можно запускать обучение!")

if not os.path.exists(teacher_path):
    print("❌ Веса учителя не найдены")
    print(f"   Ожидается: {teacher_path}")

print("\n🚀 Если все проверки пройдены, запустите:")
print("   python train.py")
print("="*60)



