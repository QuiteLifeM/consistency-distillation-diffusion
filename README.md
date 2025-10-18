# Consistency Distillation для Diffusion Models

## Что это?

Этот репозиторий содержит реализацию **Consistency Distillation** - современного метода для обучения diffusion моделей, который позволяет:

## Структура проекта

```
├── src/
│   ├── training/           # Скрипты обучения
│   │   ├── *.py           # 8 различных CD скриптов
│   │   ├── *_INFO.txt     # Информация по каждому
│   │   └── SCRIPTS_INDEX.txt
│   ├── models/            # Архитектуры моделей
│   └── utils/             # Вспомогательные функции
├── assets/
│   ├── checkpoints/       # Обученные веса моделей (8 файлов, 28GB)
│   └── images/           # Примеры генерации
├── examples/              # Демо скрипты
└── tests/                 # Тесты
```

## Основные веса моделей

| Веса модели | Размер | Описание | Скрипт |
|-------------|--------|----------|--------|
| `student_test_cd_fixed_text_encoder.pt` | 768MB | **Главные веса** (последние обученные) | `train_cd_fixed_text_encoder.py` |
| `student_final_5epochs_lr1e5.pt` | 4.4GB | Веса полного обучения 5 эпох с LR 1e-5 | `train_5_epochs_cd_lr1e5.py` |
| `student_final_hybrid.pt` | 4.4GB | Веса гибридной модели | `train_hybrid_consistency.py` |
| `student_test_cd_100_iters.pt` | 4.4GB | Веса тестовой модели на 100 итераций | `train_cd_100_iters.py` |
| `student_consistency_20iters.pt` | 828MB | Веса тестовой на 20 итераций | `test_new_consistency_20_iters.py` |

> **Скачать веса**: См. [MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md) для ссылок на Google Drive

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск обучения

```bash
# Основной скрипт 
python src/training/train_cd_fixed_text_encoder.py

# Полное обучение 5 эпох
python src/training/train_5_epochs_cd_lr1e5.py

# Тестовая модель (20 итераций)
python src/training/train_cd_20_iters.py
```

### 3. Демонстрация

```bash
# Простой тест генерации
python examples/consistency_distillation_demo.py

# Сравнение методов
python examples/methods_comparison.py
```

## Скачать обученные веса

> **Важно**: Веса моделей не включены в репозиторий из-за размера (28GB)

### Основные веса (Google Drive #1):
- **student_test_cd_fixed_text_encoder.pt** (768MB) - [Google Drive #1](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing)
- **student_final_5epochs_lr1e5.pt** (4.4GB) - [Google Drive #1](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing)  
- **student_final_hybrid.pt** (4.4GB) - [Google Drive #1](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing)
- **student_test_cd_100_iters.pt** (4.4GB) - [Google Drive #1](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing)

### Дополнительные веса (Google Drive #2):
- **student_test_cd_final.pt** (4.4GB) - [Google Drive #2](https://drive.google.com/drive/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t?usp=sharing)
- **student_test_cd_fixed_teacher.pt** (4.4GB) - [Google Drive #2](https://drive.google.com/drive/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t?usp=sharing)
- **student_test_cd_pretrained_teacher.pt** (4.4GB) - [Google Drive #2](https://drive.google.com/drive/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t?usp=sharing)
- **student_consistency_20iters.pt** (828MB) - [Google Drive #2](https://drive.google.com/drive/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t?usp=sharing)

### Быстрый старт с весами:
1. Скачайте нужные веса
2. Поместите в `assets/checkpoints/`
3. Запустите скрипт:
```bash
python src/training/train_cd_fixed_text_encoder.py
```

## Архитектуры моделей

### Student Models
- **MicroDiT_Tiny_2** - Компактная архитектура для быстрого обучения
- **MicroDiT_XL_2** - Расширенная версия с большей емкостью
- **DiT** - Стандартная Diffusion Transformer
- **create_latent_diffusion** - VAE-based архитектура

### Teacher Models
- **create_latent_diffusion** - Основной учитель для большинства экспериментов
- **SDXL Turbo** - Для специализированных задач

## Конфигурации

| Параметр | Значения | Описание |
|----------|----------|----------|
| **Batch Size** | 1, 2, 4 | Оптимизировано для разных GPU |
| **Learning Rate** | 1e-4, 1e-5 | Различные скорости обучения |
| **Epochs** | 1-5 | От быстрого теста до полного обучения |
| **Iterations** | 20-4000 | Гибкая настройка длительности |

## Примеры использования

### Базовое обучение
```python
from src.training.train_cd_fixed_text_encoder import train_consistency_distillation

# Запуск с настройками по умолчанию
model = train_consistency_distillation()
```

### Кастомная конфигурация
```python
# Изменение batch size для вашей GPU
BATCH_SIZE = 2  # Для GPU с 8GB памяти
LEARNING_RATE = 1e-5  # Более консервативное обучение
```


