import os
import torch
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion

# Настройка для CPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def save_models():
    """Сохраняем готовые модели для быстрой загрузки"""
    print("🔄 Создание и сохранение моделей...")
    
    # =======================
    # Создание модели учителя
    # =======================
    print("Создание модели учителя...")
    try:
        teacher_model = create_latent_diffusion(
            latent_res=64,
            in_channels=4,
            pos_interp_scale=2.0,
            precomputed_latents=False,
            dtype="float32"
        )
        print("✅ Модель учителя создана")
    except Exception as e:
        print(f"❌ Ошибка при создании модели учителя: {e}")
        return False

    # Загружаем веса учителя
    print("Загружаем веса учителя...")
    try:
        teacher_weights = torch.load("./micro_diffusion/trained_models/teacher.pt", map_location="cpu")
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        teacher_model.eval()
        teacher_model = teacher_model.to("cpu")
        print("✅ Веса учителя загружены")
    except Exception as e:
        print(f"❌ Ошибка при загрузке весов учителя: {e}")
        return False

    # =======================
    # Создание модели студента
    # =======================
    print("Создание модели студента...")
    try:
        student_model = create_latent_diffusion(
            latent_res=64,
            in_channels=4,
            pos_interp_scale=2.0,
            precomputed_latents=False,
            dtype="float32"
        )
        print("✅ Модель студента создана")
    except Exception as e:
        print(f"❌ Ошибка при создании модели студента: {e}")
        return False

    # Инициализируем студента весами учителя
    print("Инициализация студента весами учителя...")
    try:
        student_model.dit.load_state_dict(teacher_weights, strict=False)
        student_model.train()
        student_model = student_model.to("cpu")
        print("✅ Студент инициализирован весами учителя")
    except Exception as e:
        print(f"❌ Ошибка при инициализации студента: {e}")
        return False

    # =======================
    # Сохранение моделей
    # =======================
    print("Сохраняем модели...")
    try:
        # Сохраняем полные модели
        torch.save(teacher_model, "teacher_model_ready.pt")
        print("✅ Учитель сохранен: teacher_model_ready.pt")
        
        torch.save(student_model, "student_model_ready.pt")
        print("✅ Студент сохранен: student_model_ready.pt")
        
        # Сохраняем веса отдельно для совместимости
        torch.save(teacher_weights, "teacher_weights.pt")
        print("✅ Веса учителя сохранены: teacher_weights.pt")
        
        print("\n🎉 Все модели сохранены успешно!")
        print("Теперь можно использовать train_fast_load.py для быстрой загрузки")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при сохранении моделей: {e}")
        return False

if __name__ == "__main__":
    print("💾 Сохранение моделей для быстрой загрузки")
    print("⏱️  Это займет 5-10 минут, но потом будет очень быстро!")
    
    success = save_models()
    
    if success:
        print("\n✅ Готово! Теперь можно запускать:")
        print("   python train_fast_load.py")
    else:
        print("\n❌ Ошибка при сохранении моделей")

