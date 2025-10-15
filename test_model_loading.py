"""
Тест загрузки модели с минимальными параметрами
"""
import torch
import os

def test_cuda():
    """Проверка CUDA"""
    print("="*60)
    print("Проверка CUDA")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна")
        return False
    
    print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Свободно: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    return True

def test_imports():
    """Проверка импортов"""
    print("\n" + "="*60)
    print("Проверка импортов")
    print("="*60)
    
    try:
        print("1. Импорт torch...")
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        print("2. Импорт micro_diffusion...")
        from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
        print("   ✅ micro_diffusion импортирован")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка импорта: {e}")
        return False

def test_model_creation():
    """Тест создания модели с минимальными параметрами"""
    print("\n" + "="*60)
    print("Тест создания модели")
    print("="*60)
    
    try:
        from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
        
        print("1. Создание модели на CPU...")
        model = create_latent_diffusion(
            latent_res=32,  # Меньший размер
            in_channels=4,
            pos_interp_scale=1.0,  # Меньший scale
            precomputed_latents=False,
            dtype="float32"
        )
        print("   ✅ Модель создана на CPU")
        
        print("2. Проверка компонентов модели...")
        print(f"   - DiT: {type(model.dit)}")
        print(f"   - VAE: {type(model.vae)}")
        print(f"   - Text Encoder: {type(model.text_encoder)}")
        print(f"   - Tokenizer: {type(model.tokenizer)}")
        
        print("3. Попытка перемещения на GPU...")
        try:
            model = model.to("cuda")
            print("   ✅ Модель перемещена на GPU")
        except Exception as e:
            print(f"   ⚠️ Не удалось переместить на GPU: {e}")
            print("   Продолжаем с CPU")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка создания модели: {e}")
        return False

def test_teacher_weights():
    """Тест загрузки весов учителя"""
    print("\n" + "="*60)
    print("Тест загрузки весов учителя")
    print("="*60)
    
    teacher_path = "./micro_diffusion/trained_models/teacher.pt"
    
    if not os.path.exists(teacher_path):
        print(f"❌ Файл весов не найден: {teacher_path}")
        return False
    
    try:
        print("1. Загрузка весов...")
        weights = torch.load(teacher_path, map_location="cpu")
        print(f"   ✅ Веса загружены, размер: {os.path.getsize(teacher_path) / 1024**2:.1f} MB")
        
        print("2. Анализ структуры весов...")
        print(f"   Ключей в весах: {len(weights)}")
        
        # Показываем первые несколько ключей
        keys = list(weights.keys())[:5]
        for key in keys:
            shape = weights[key].shape if hasattr(weights[key], 'shape') else 'N/A'
            print(f"   - {key}: {shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка загрузки весов: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Диагностика проблем с загрузкой модели")
    print("="*60)
    
    # Тест 1: CUDA
    cuda_ok = test_cuda()
    
    # Тест 2: Импорты
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать модули")
        exit(1)
    
    # Тест 3: Создание модели
    model_ok = test_model_creation()
    
    # Тест 4: Веса учителя
    weights_ok = test_teacher_weights()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    print("="*60)
    print(f"CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"Импорты: {'✅' if imports_ok else '❌'}")
    print(f"Создание модели: {'✅' if model_ok else '❌'}")
    print(f"Веса учителя: {'✅' if weights_ok else '❌'}")
    
    if cuda_ok and imports_ok and model_ok and weights_ok:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Модель должна работать.")
        print("Попробуйте запустить: python train.py")
    else:
        print("\n⚠️ ЕСТЬ ПРОБЛЕМЫ. Рекомендации:")
        if not cuda_ok:
            print("- Проверьте установку CUDA и PyTorch")
        if not model_ok:
            print("- Попробуйте уменьшить latent_res до 32")
            print("- Проверьте совместимость версий")
        if not weights_ok:
            print("- Проверьте путь к файлу teacher.pt")
    
    print("="*60)



