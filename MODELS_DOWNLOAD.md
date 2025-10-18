# Скачать обученные веса

> **Важно**: Веса моделей не включены в репозиторий из-за размера (28GB)

## Основные веса моделей

| Веса модели | Размер | Описание | Ссылка |
|-------------|--------|----------|--------|
| `student_test_cd_fixed_text_encoder.pt` | 768MB | **Главные веса** (последние обученные) | [Google Drive](https://drive.google.com/drive/u/2/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d) |
| `student_final_5epochs_lr1e5.pt` | 4.4GB | Веса полного обучения 5 эпох с LR 1e-5 | [Google Drive](https://drive.google.com/drive/u/3/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t) |
| `student_final_hybrid.pt` | 4.4GB | Веса гибридной модели | [Google Drive](https://drive.google.com/drive/u/3/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t) |
| `student_test_cd_100_iters.pt` | 4.4GB | Веса тестовой модели на 100 итераций | [Google Drive](https://drive.google.com/drive/u/3/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t) |
| `student_test_cd_final.pt` | 4.4GB | Веса финальной тестовой модели | [Google Drive](https://drive.google.com/drive/u/2/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d) |
| `student_test_cd_fixed_teacher.pt` | 4.4GB | Веса с исправленным учителем | [Google Drive](https://drive.google.com/drive/u/2/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d) |
| `student_test_cd_pretrained_teacher.pt` | 4.4GB | Веса с предобученным учителем | [Google Drive](https://drive.google.com/drive/u/2/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d) |
| `student_consistency_20iters.pt` | 828MB | Веса тестовой на 20 итераций | [Google Drive](https://drive.google.com/drive/u/3/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t) |

## Все веса (28GB)

**Полные архивы**: 
- [Google Drive - Папка 1](https://drive.google.com/drive/u/2/folders/1UIpo6Ac-UimM03qLn6Ty6g4D56GRo21d) (4 файла)
- [Google Drive - Папка 2](https://drive.google.com/drive/u/3/folders/14Frua7p6ZejptuRrXo_O9dEFwIYtdi0t) (4 файла)

## Быстрый старт

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

## Размеры файлов

- **Общий размер весов**: 28GB
- **Размер репозитория**: ~5MB (только код)
- **Экономия места**: 99.98%

## Troubleshooting

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
