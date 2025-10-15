"""
Визуализация прогресса обучения
"""
import matplotlib.pyplot as plt
import torch
import os
import glob

def plot_loss_curve(loss_history=None, save_path="training_loss.png"):
    """
    Построение графика loss'а
    """
    if loss_history is None:
        print("⚠️ История loss'ов не передана")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 
             marker='o', linewidth=2, markersize=6, color='blue')
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Consistency Distillation Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ График сохранен: {save_path}")
    plt.close()


def analyze_checkpoints():
    """
    Анализ сохраненных чекпоинтов
    """
    print("="*60)
    print("Анализ чекпоинтов")
    print("="*60)
    
    checkpoint_files = sorted(glob.glob("student_checkpoint_epoch_*.pt"))
    
    if not checkpoint_files:
        print("⚠️ Чекпоинты не найдены")
        return
    
    print(f"\nНайдено {len(checkpoint_files)} чекпоинтов:")
    
    for cp_file in checkpoint_files:
        # Загружаем чекпоинт
        checkpoint = torch.load(cp_file, map_location="cpu")
        
        # Размер файла
        file_size = os.path.getsize(cp_file) / (1024**2)  # MB
        
        # Количество параметров
        num_params = sum(p.numel() for p in checkpoint.values())
        
        print(f"\n  📁 {cp_file}")
        print(f"     Размер: {file_size:.1f} MB")
        print(f"     Параметров: {num_params:,}")
    
    # Финальная модель
    if os.path.exists("student_final.pt"):
        final_checkpoint = torch.load("student_final.pt", map_location="cpu")
        file_size = os.path.getsize("student_final.pt") / (1024**2)
        num_params = sum(p.numel() for p in final_checkpoint.values())
        
        print(f"\n  📁 student_final.pt (финальная модель)")
        print(f"     Размер: {file_size:.1f} MB")
        print(f"     Параметров: {num_params:,}")
    
    print("\n" + "="*60)


def compare_model_sizes():
    """
    Сравнение размеров учителя и студента
    """
    print("="*60)
    print("Сравнение размеров моделей")
    print("="*60)
    
    teacher_path = "./micro_diffusion/trained_models/teacher.pt"
    student_path = "student_final.pt"
    
    if os.path.exists(teacher_path):
        teacher_size = os.path.getsize(teacher_path) / (1024**2)
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu")
        teacher_params = sum(p.numel() for p in teacher_checkpoint.values())
        
        print(f"\n👨‍🏫 Учитель (teacher.pt):")
        print(f"   Размер: {teacher_size:.1f} MB")
        print(f"   Параметров: {teacher_params:,}")
    else:
        print(f"\n⚠️ Учитель не найден: {teacher_path}")
    
    if os.path.exists(student_path):
        student_size = os.path.getsize(student_path) / (1024**2)
        student_checkpoint = torch.load(student_path, map_location="cpu")
        student_params = sum(p.numel() for p in student_checkpoint.values())
        
        print(f"\n👨‍🎓 Студент (student_final.pt):")
        print(f"   Размер: {student_size:.1f} MB")
        print(f"   Параметров: {student_params:,}")
        
        if os.path.exists(teacher_path):
            print(f"\n📊 Сравнение:")
            print(f"   Размер: одинаковый ({student_size:.1f} MB)")
            print(f"   Параметров: одинаковое ({student_params:,})")
            print(f"   Преимущество: студент генерирует быстрее (меньше шагов)")
    else:
        print(f"\n⚠️ Студент не найден: {student_path}")
        print("   Запустите train.py для обучения модели")
    
    print("\n" + "="*60)


def plot_multiple_metrics(metrics_dict, save_path="training_metrics.png"):
    """
    Построение нескольких метрик на одном графике
    
    Args:
        metrics_dict: словарь {название: [значения по эпохам]}
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in metrics_dict.items():
        plt.plot(range(1, len(values) + 1), values, 
                marker='o', linewidth=2, label=name)
    
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.title('Метрики обучения', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ График метрик сохранен: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Визуализация результатов обучения")
    parser.add_argument("--mode", choices=["checkpoints", "compare", "all"], 
                        default="all",
                        help="Режим анализа")
    
    args = parser.parse_args()
    
    if args.mode in ["checkpoints", "all"]:
        analyze_checkpoints()
    
    if args.mode in ["compare", "all"]:
        compare_model_sizes()
    
    print("\n💡 Совет: Для визуализации loss'а используйте график training_loss.png,")
    print("   который создается автоматически после обучения.")




