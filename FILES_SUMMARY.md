# Созданные файлы - Consistency Distillation

Полный список всех созданных и обновленных файлов для проекта Consistency Distillation.

## 📝 Основные файлы

### 1. `train.py` ⭐ (ОБНОВЛЕН)
**Назначение:** Главный скрипт обучения модели студента через Consistency Distillation

**Содержит:**
- `LatentPromptDataset` - класс датасета для загрузки латентов и промптов
- `get_text_embeddings()` - получение текстовых эмбеддингов через модель
- `consistency_distillation_step()` - один шаг CD обучения
- `custom_collate()` - обработка батчей
- `train_consistency_distillation()` - основной цикл обучения
- Автоматическое сохранение чекпоинтов и графика

**Использование:**
```bash
python train.py
```

**Выходные файлы:**
- `student_checkpoint_epoch_*.pt` - чекпоинты каждые 2 эпохи
- `student_final.pt` - финальная обученная модель
- `training_loss.png` - график обучения

---

### 2. `test_data_loading.py` 🧪
**Назначение:** Тестирование загрузки данных перед обучением

**Проверяет:**
- Корректность путей к латентам и промптам
- Размерность и тип данных латентов
- Работу DataLoader с батчами

**Использование:**
```bash
python test_data_loading.py
```

**Ожидаемый вывод:**
```
✅ Загружено 4000 латентов и промптов
Latent shape: torch.Size([4, 64, 64])
...
✅ All tests passed!
```

---

### 3. `generate_with_student.py` 🎨
**Назначение:** Генерация изображений с помощью обученной модели студента

**Режимы работы:**

#### Режим `generate` (по умолчанию)
Генерирует несколько изображений с дефолтными промптами:
```bash
python generate_with_student.py --mode generate
```

Параметры:
- `--steps` - количество шагов денойзинга (по умолчанию: 10)
- `--cfg` - guidance scale (по умолчанию: 5.0)
- `--seed` - random seed (по умолчанию: 42)
- `--model` - путь к весам студента (по умолчанию: student_final.pt)

#### Режим `compare`
Сравнивает генерацию учителя и студента:
```bash
python generate_with_student.py --mode compare --prompt "your prompt here"
```

**Выходные файлы:**
- `generated_01.png`, `generated_02.png`, ... (режим generate)
- `comparison_teacher.png`, `comparison_student.png` (режим compare)

---

### 4. `visualize_training.py` 📊
**Назначение:** Визуализация и анализ результатов обучения

**Функции:**
- `analyze_checkpoints()` - анализ сохраненных чекпоинтов
- `compare_model_sizes()` - сравнение размеров учителя и студента
- `plot_loss_curve()` - построение графика loss'а
- `plot_multiple_metrics()` - графики нескольких метрик

**Использование:**
```bash
# Анализ всего
python visualize_training.py --mode all

# Только чекпоинты
python visualize_training.py --mode checkpoints

# Только сравнение моделей
python visualize_training.py --mode compare
```

---

## 📚 Документация

### 5. `CONSISTENCY_DISTILLATION_README.md` 📖
**Назначение:** Подробная документация проекта

**Содержит:**
- Что такое Consistency Distillation
- Структура проекта
- Как работает алгоритм
- Параметры для настройки
- Ожидаемые результаты
- Troubleshooting
- Дополнительные улучшения

**Для кого:** Подробное изучение алгоритма и архитектуры

---

### 6. `QUICKSTART.md` 🚀
**Назначение:** Быстрое руководство по запуску

**Содержит:**
- Пошаговая инструкция (4 шага)
- Команды для запуска
- Ожидаемые выводы
- Частые проблемы и решения
- Примеры использования

**Для кого:** Быстрый старт, если нужно сразу запустить

---

### 7. `FILES_SUMMARY.md` 📋 (этот файл)
**Назначение:** Обзор всех файлов проекта

---

## 📂 Структура проекта

```
train/
│
├── 📜 Основные скрипты
│   ├── train.py                     ⭐ Главный скрипт обучения
│   ├── test_data_loading.py         🧪 Тестирование данных
│   ├── generate_with_student.py     🎨 Генерация изображений
│   └── visualize_training.py        📊 Визуализация результатов
│
├── 📚 Документация
│   ├── QUICKSTART.md                🚀 Быстрый старт
│   ├── CONSISTENCY_DISTILLATION_README.md  📖 Подробная документация
│   └── FILES_SUMMARY.md             📋 Этот файл
│
├── 📁 Данные
│   └── datadir/
│       ├── latents/                 (4000 .pt файлов)
│       └── prompts/                 (4000 .txt файлов)
│
├── 🤖 Модель
│   └── micro_diffusion/
│       ├── micro_diffusion/         (код модели)
│       └── trained_models/
│           └── teacher.pt           (веса учителя)
│
└── 📦 Выходные файлы (создаются после обучения)
    ├── student_checkpoint_epoch_*.pt
    ├── student_final.pt
    ├── training_loss.png
    ├── generated_*.png
    └── comparison_*.png
```

---

## 🔄 Workflow

### 1️⃣ Подготовка
```bash
python test_data_loading.py
```
✅ Проверяем, что данные загружаются корректно

### 2️⃣ Обучение
```bash
python train.py
```
✅ Обучаем студента (~1-2 часа)

### 3️⃣ Анализ
```bash
python visualize_training.py --mode all
```
✅ Анализируем результаты обучения

### 4️⃣ Генерация
```bash
python generate_with_student.py --mode compare
```
✅ Тестируем обученную модель

---

## 🎯 Быстрая навигация

| Задача | Файл |
|--------|------|
| Хочу быстро запустить | `QUICKSTART.md` |
| Хочу понять, как это работает | `CONSISTENCY_DISTILLATION_README.md` |
| Хочу начать обучение | `train.py` |
| Хочу проверить данные | `test_data_loading.py` |
| Хочу сгенерировать изображения | `generate_with_student.py` |
| Хочу посмотреть метрики | `visualize_training.py` |
| Хочу понять структуру проекта | `FILES_SUMMARY.md` (этот файл) |

---

## 💡 Советы

1. **Начните с QUICKSTART.md** если хотите быстро запустить
2. **Всегда запускайте test_data_loading.py** перед обучением
3. **Уменьшите batch_size** если видите CUDA out of memory
4. **Сохраняйте чекпоинты** - они занимают место, но полезны
5. **Экспериментируйте с параметрами** в generate_with_student.py

---

## 📊 Метрики успеха

После обучения проверьте:

✅ Loss уменьшается (см. `training_loss.png`)  
✅ Модель студента сгенерирована (`student_final.pt`)  
✅ Генерация работает (`python generate_with_student.py`)  
✅ Качество изображений сопоставимо с учителем  
✅ Скорость генерации выше (меньше шагов)  

---

## 🔗 Связь между файлами

```
train.py
   ↓ создает
student_final.pt
   ↓ использует
generate_with_student.py
   ↓ создает
generated_*.png

train.py
   ↓ создает
training_loss.png, checkpoints
   ↓ анализирует
visualize_training.py
```

---

## 🛠️ Кастомизация

### Изменить размер батча
`train.py`, строка 262:
```python
batch_size=4  # измените на 2 или 1
```

### Изменить количество эпох
`train.py`, строка 270:
```python
num_epochs = 10  # измените на нужное
```

### Изменить learning rate
`train.py`, строка 74:
```python
lr=1e-5  # измените на 1e-6 для более медленного обучения
```

### Изменить количество шагов генерации
`generate_with_student.py`:
```bash
python generate_with_student.py --steps 15  # вместо 10
```

---

**Версия:** 1.0  
**Дата:** 2025  
**Статус:** ✅ Готово к использованию




