"""
Скрипт для генерации изображений с помощью обученной студенческой модели
"""
import torch
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
from PIL import Image
import numpy as np

def generate_images(
    model_path="student_final.pt",
    prompts=None,
    num_inference_steps=10,
    guidance_scale=5.0,
    seed=42,
    output_prefix="generated"
):
    """
    Генерация изображений с помощью обученной модели студента
    
    Args:
        model_path: путь к весам обученной модели
        prompts: список промптов (если None, используются примеры)
        num_inference_steps: количество шагов денойзинга (меньше = быстрее)
        guidance_scale: сила следования промпту (обычно 5-10)
        seed: random seed для воспроизводимости
        output_prefix: префикс для сохраненных файлов
    """
    
    if prompts is None:
        prompts = [
            "a beautiful landscape with mountains and lake",
            "a cute cat sitting on a windowsill",
            "a futuristic city at night",
            "a portrait of a person in oil painting style"
        ]
    
    print("="*60)
    print("Генерация изображений с помощью Consistency-Distilled модели")
    print("="*60)
    
    # Загружаем модель студента
    print("\n1. Загрузка модели студента...")
    student_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cuda")
    
    # Загружаем обученные веса
    print(f"2. Загрузка весов из {model_path}...")
    student_model.dit.load_state_dict(
        torch.load(model_path, map_location="cuda")
    )
    student_model.eval()
    print("✅ Модель загружена и готова к генерации")
    
    # Генерируем изображения
    print(f"\n3. Генерация {len(prompts)} изображений...")
    print(f"   Параметры: steps={num_inference_steps}, cfg={guidance_scale}, seed={seed}")
    
    for i, prompt in enumerate(prompts):
        print(f"\n   [{i+1}/{len(prompts)}] Генерация: '{prompt}'")
        
        # Генерация
        with torch.no_grad():
            image_tensor = student_model.generate(
                prompt=[prompt],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed + i,  # Разные seed для каждого изображения
                return_only_latents=False
            )
        
        # Конвертация в PIL Image
        image_np = image_tensor[0].cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8).transpose(1, 2, 0)
        image_pil = Image.fromarray(image_np)
        
        # Сохранение
        output_path = f"{output_prefix}_{i+1:02d}.png"
        image_pil.save(output_path)
        print(f"   ✅ Сохранено: {output_path}")
    
    print("\n" + "="*60)
    print("✅ Все изображения сгенерированы!")
    print("="*60)


def compare_teacher_student(
    teacher_path="./micro_diffusion/trained_models/teacher.pt",
    student_path="student_final.pt",
    prompt="a beautiful landscape with mountains",
    num_steps_teacher=30,
    num_steps_student=10,
    guidance_scale=5.0,
    seed=42
):
    """
    Сравнение генерации учителя и студента
    """
    print("="*60)
    print("Сравнение Teacher vs Student")
    print("="*60)
    
    # Загружаем учителя
    print("\n1. Загрузка учителя...")
    teacher = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cuda")
    teacher.dit.load_state_dict(torch.load(teacher_path, map_location="cuda"))
    teacher.eval()
    print(f"✅ Учитель загружен (будет использовать {num_steps_teacher} шагов)")
    
    # Загружаем студента
    print("\n2. Загрузка студента...")
    student = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cuda")
    student.dit.load_state_dict(torch.load(student_path, map_location="cuda"))
    student.eval()
    print(f"✅ Студент загружен (будет использовать {num_steps_student} шагов)")
    
    # Генерация учителем
    print(f"\n3. Генерация учителем (prompt: '{prompt}')...")
    import time
    start = time.time()
    with torch.no_grad():
        teacher_image = teacher.generate(
            prompt=[prompt],
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps_teacher,
            seed=seed,
            return_only_latents=False
        )
    teacher_time = time.time() - start
    print(f"   ✅ Время: {teacher_time:.2f}s")
    
    # Генерация студентом
    print(f"\n4. Генерация студентом (prompt: '{prompt}')...")
    start = time.time()
    with torch.no_grad():
        student_image = student.generate(
            prompt=[prompt],
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps_student,
            seed=seed,
            return_only_latents=False
        )
    student_time = time.time() - start
    print(f"   ✅ Время: {student_time:.2f}s")
    
    # Сохранение результатов
    print("\n5. Сохранение результатов...")
    
    # Teacher
    teacher_np = (teacher_image[0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(teacher_np).save("comparison_teacher.png")
    print(f"   ✅ Учитель: comparison_teacher.png")
    
    # Student
    student_np = (student_image[0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(student_np).save("comparison_student.png")
    print(f"   ✅ Студент: comparison_student.png")
    
    # Статистика
    print("\n" + "="*60)
    print("Результаты сравнения:")
    print(f"  Учитель: {num_steps_teacher} шагов, {teacher_time:.2f}s")
    print(f"  Студент: {num_steps_student} шагов, {student_time:.2f}s")
    print(f"  Ускорение: {teacher_time/student_time:.2f}x")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Генерация с помощью обученной студенческой модели")
    parser.add_argument("--mode", choices=["generate", "compare"], default="generate",
                        help="Режим: generate (генерация) или compare (сравнение с учителем)")
    parser.add_argument("--model", default="student_final.pt",
                        help="Путь к весам студента")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Промпт для генерации (только для режима compare)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Количество шагов для студента")
    parser.add_argument("--cfg", type=float, default=5.0,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_images(
            model_path=args.model,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )
    else:
        prompt = args.prompt if args.prompt else "a beautiful landscape with mountains"
        compare_teacher_student(
            student_path=args.model,
            prompt=prompt,
            num_steps_student=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )




