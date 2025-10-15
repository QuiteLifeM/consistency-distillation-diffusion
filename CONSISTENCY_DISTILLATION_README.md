# Consistency Distillation для micro_diffusion

Этот проект реализует Consistency Distillation для дистилляции модели micro_diffusion.

## Что такое Consistency Distillation?

Consistency Distillation - это метод обучения, который позволяет создать "студенческую" модель, которая может генерировать изображения за меньшее количество шагов, чем оригинальная "учительская" модель. Студент обучается предсказывать тот же результат денойзинга, что и учитель, для любого уровня шума.

## Структура проекта

```
train/
├── datadir/
│   ├── latents/          # 4000 предвычисленных латентов (.pt файлы)
│   └── prompts/          # 4000 текстовых промптов (.txt файлы)
├── micro_diffusion/      # Код модели micro_diffusion
│   └── trained_models/
│       └── teacher.pt    # Веса учителя
├── train.py              # Основной скрипт обучения
├── test_data_loading.py  # Тест загрузки данных
└── CONSISTENCY_DISTILLATION_README.md  # Этот файл
```

## Требования

- Python 3.8+
- PyTorch 2.0+
- CUDA-совместимая GPU (минимум 8 GB VRAM)
- Установленный пакет micro_diffusion

## Как работает скрипт train.py

### 1. Загрузка данных

```python
class LatentPromptDataset(Dataset):
    """
    Загружает предвычисленные латенты (4, 64, 64) и текстовые промпты
    """
```

- Латенты: 4-канальные тензоры размера 64x64 (результат VAE энкодинга 512x512 изображений)
- Промпты: текстовые описания изображений

### 2. Модели

**Учитель (Teacher Model):**
- Загружается с предобученными весами из `teacher.pt`
- Всегда в режиме `eval()` (не обучается)
- Генерирует "целевые" предсказания для студента

**Студент (Student Model):**
- Инициализируется весами учителя
- Обучается предсказывать те же результаты, что и учитель
- В режиме `train()`

### 3. Consistency Distillation Step

Для каждого батча:
1. Берем чистый латент изображения
2. Добавляем случайный шум (сэмплированный из EDM распределения)
3. Учитель делает один шаг денойзинга → получаем `teacher_denoised`
4. Студент делает один шаг денойзинга → получаем `student_denoised`
5. Вычисляем MSE loss между предсказаниями: `loss = MSE(student_denoised, teacher_denoised)`
6. Обновляем веса студента через backpropagation

### 4. Обучение

```python
train_consistency_distillation(
    dataloader, 
    teacher_model, 
    student_model, 
    optimizer, 
    num_epochs=10
)
```

- Логирование каждые 10 батчей
- Сохранение чекпоинтов каждые 2 эпохи
- Финальное сохранение модели студента
- График loss'а

## Запуск обучения

### Шаг 1: Проверка данных

```bash
python test_data_loading.py
```

Должно вывести:
```
✅ Загружено 4000 латентов и промптов
Latent shape: torch.Size([4, 64, 64])
Latent dtype: torch.float32
...
```

### Шаг 2: Запуск обучения

```bash
python train.py
```

### Что произойдет:
1. Загрузка учителя и студента (~30 сек)
2. Создание DataLoader
3. Обучение 10 эпох (~1-2 часа на 4000 сэмплов с batch_size=4)
4. Сохранение чекпоинтов: `student_checkpoint_epoch_2.pt`, `student_checkpoint_epoch_4.pt`, и т.д.
5. Сохранение финальной модели: `student_final.pt`
6. Создание графика: `training_loss.png`

## Параметры для настройки

В `train.py` можно изменить:

```python
# Размер батча (меньше = меньше памяти, но медленнее)
batch_size = 4

# Количество эпох
num_epochs = 10

# Learning rate
lr = 1e-5

# Частота логирования
if i % 10 == 0:  # Можно изменить на 50, 100, и т.д.
```

## Использование обученной модели

После обучения вы можете загрузить студента для генерации:

```python
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion
import torch

# Создаем модель студента
student_model = create_latent_diffusion(
    latent_res=64,
    in_channels=4,
    pos_interp_scale=2.0,
    precomputed_latents=False,
    dtype="float32"
).to("cuda")

# Загружаем обученные веса
student_model.dit.load_state_dict(
    torch.load("student_final.pt", map_location="cuda")
)

# Генерация изображения
image = student_model.generate(
    prompt=["a beautiful landscape"],
    guidance_scale=5.0,
    num_inference_steps=10,  # Меньше шагов чем у учителя!
    seed=42
)
```

## Ожидаемые результаты

- **Loss:** Должен уменьшаться с каждой эпохой (от ~0.1-0.5 до ~0.01-0.05)
- **Качество:** Студент должен генерировать изображения сравнимого качества с учителем, но за меньшее количество шагов
- **Скорость:** Генерация в 2-4 раза быстрее

## Troubleshooting

### Ошибка: "CUDA out of memory"
- Уменьшите `batch_size` (попробуйте 2 или 1)
- Используйте `torch.cuda.empty_cache()` между эпохами

### Ошибка: "Input height (32) doesn't match model (64)"
- Проверьте, что латенты имеют размер [4, 64, 64]
- Убедитесь, что используете правильный `latent_res=64` для 512px изображений

### Loss не уменьшается
- Попробуйте уменьшить learning rate (например, `1e-6`)
- Убедитесь, что учитель загружен правильно
- Проверьте, что студент начинается с весов учителя

### Медленная скорость обучения
- Используйте `num_workers=2` в DataLoader (но может быть нестабильно на Windows)
- Уменьшите частоту логирования

## Дополнительные улучшения

Возможные улучшения алгоритма:

1. **Multi-step Consistency:** Учить студента делать несколько шагов денойзинга
2. **EMA (Exponential Moving Average):** Использовать EMA весов для стабильности
3. **Progressive Distillation:** Постепенно уменьшать количество шагов
4. **Curriculum Learning:** Начинать с простых уровней шума, постепенно усложняя

## Ссылки

- [Consistency Models Paper](https://arxiv.org/abs/2303.01469)
- [Progressive Distillation](https://arxiv.org/abs/2202.00512)
- [micro_diffusion Documentation](https://github.com/...)

---

Удачи в обучении! 🚀




