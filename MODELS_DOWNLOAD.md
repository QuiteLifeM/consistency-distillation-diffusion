# 📥 Скачать обученные веса

> **Важно**: Веса моделей не включены в репозиторий из-за размера (28GB)

## 🎯 Основные веса моделей

| Веса модели | Размер | Описание | Ссылка |
|-------------|--------|----------|--------|
| `student_test_cd_fixed_text_encoder.pt` | 1.2GB | **Главные веса** (последние обученные) | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_final_5epochs_lr1e5.pt` | 1.2GB | Веса полного обучения 5 эпох с LR 1e-5 | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_final_hybrid.pt` | 1.2GB | Веса гибридной модели | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_test_cd_100_iters.pt` | 1.2GB | Веса тестовой модели на 100 итераций | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_test_cd_final.pt` | 1.2GB | Веса финальной тестовой модели | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_test_cd_fixed_teacher.pt` | 1.2GB | Веса с исправленным учителем | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_test_cd_pretrained_teacher.pt` | 1.2GB | Веса с предобученным учителем | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |
| `student_consistency_20iters.pt` | 1.2GB | Веса тестовой на 20 итераций | [Google Drive](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing) |

## 📦 Все веса (28GB)

**Полный архив**: 
- [Google Drive - Все веса](https://drive.google.com/drive/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d?usp=sharing)

## 🚀 Быстрый старт

### 1. Скачайте нужные веса
Выберите веса для вашей задачи:
- **Для демонстрации**: `student_test_cd_fixed_text_encoder.pt`
- **Для полного обучения**: `student_final_5epochs_lr1e5.pt`
- **Для тестирования**: `student_consistency_20iters.pt`

### 2. Поместите в папку
```bash
# Создайте папку для весов
mkdir -p assets/checkpoints/

# Скопируйте скачанные файлы
cp ~/Downloads/student_*.pt assets/checkpoints/
```

### 3. Запустите скрипт
```bash
# Основной скрипт
python src/training/train_cd_fixed_text_encoder.py

# Тестовая модель
python src/training/test_new_consistency_20_iters.py
```

## 📊 Размеры файлов

- **Общий размер весов**: 28GB
- **Размер репозитория**: ~5MB (только код)
- **Экономия места**: 99.98%

## 🔧 Troubleshooting

### Ошибка "Model not found"
```bash
# Проверьте наличие весов
ls -la assets/checkpoints/

# Должны быть файлы .pt
```

### Ошибка "CUDA out of memory"
```bash
# Используйте CPU версию
python src/training/train_cd_cpu_offload.py
```
