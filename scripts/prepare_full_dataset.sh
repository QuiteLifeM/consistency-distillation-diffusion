#!/bin/bash

# Полная подготовка датасета для Consistency Distillation
# 1. Генерация 4000 изображений с SDXL-Turbo
# 2. Конвертация в латенты
# 3. Готово к обучению!

set -e  # Выход при ошибке

echo "=========================================="
echo "🚀 ПОДГОТОВКА ДАТАСЕТА ДЛЯ CONSISTENCY DISTILLATION"
echo "=========================================="

# Шаг 1: Установка зависимостей
echo ""
echo "📦 Шаг 1: Проверка зависимостей..."
pip3 install diffusers transformers accelerate --upgrade -q
echo "✅ Зависимости установлены"

# Шаг 2: Генерация изображений с SDXL-Turbo
echo ""
echo "🎨 Шаг 2: Генерация 4000 изображений с SDXL-Turbo..."
echo "   (Это займет ~2-3 часа на RTX 3090)"
echo "=========================================="
python3 generate_dataset_sdxl_turbo.py \
    --num_images 4000 \
    --output_dir dataset_sdxl_turbo

# Проверяем результат
if [ ! -d "dataset_sdxl_turbo" ]; then
    echo "❌ Ошибка: dataset_sdxl_turbo не создана!"
    exit 1
fi

IMG_COUNT=$(ls dataset_sdxl_turbo/*.png 2>/dev/null | wc -l)
echo "✅ Сгенерировано изображений: $IMG_COUNT"

# Шаг 3: Конвертация в латенты
echo ""
echo "🔄 Шаг 3: Конвертация изображений в латенты..."
echo "   (Это займет ~30-60 минут)"
echo "=========================================="

# Создаем резервные копии старых данных (если есть)
if [ -d "datadir/latents_good" ]; then
    echo "📦 Создание резервной копии старых латентов..."
    mv datadir/latents_good datadir/latents_good_backup_$(date +%Y%m%d_%H%M%S)
fi

if [ -d "datadir/prompts_good" ]; then
    echo "📦 Создание резервной копии старых промптов..."
    mv datadir/prompts_good datadir/prompts_good_backup_$(date +%Y%m%d_%H%M%S)
fi

python3 convert_images_to_latents.py \
    --images_dir dataset_sdxl_turbo \
    --output_latents_dir datadir/latents_good \
    --output_prompts_dir datadir/prompts_good

# Проверяем результат
LATENT_COUNT=$(ls datadir/latents_good/*.pt 2>/dev/null | wc -l)
PROMPT_COUNT=$(ls datadir/prompts_good/*.txt 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "✅ ПОДГОТОВКА ЗАВЕРШЕНА!"
echo "=========================================="
echo "📊 Статистика:"
echo "   🖼️  Изображений: $IMG_COUNT"
echo "   🧠 Латентов: $LATENT_COUNT"
echo "   📝 Промптов: $PROMPT_COUNT"
echo ""
echo "📁 Данные готовы:"
echo "   • Исходные изображения: dataset_sdxl_turbo/"
echo "   • Латенты: datadir/latents_good/"
echo "   • Промпты: datadir/prompts_good/"
echo ""
echo "🚀 Теперь можно запустить обучение:"
echo "   python3 train_true_consistency_distillation.py"
echo "=========================================="





