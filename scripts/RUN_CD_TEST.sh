#!/bin/bash
# 🚀 Скрипт для запуска тестового обучения Consistency Distillation

echo "🧪 Запуск тестового обучения True Consistency Distillation"
echo "========================================================================"
echo ""
echo "✅ Исправления применены:"
echo "   • text_embeddings: [B, 1, seq, dim] -> [B, seq, dim] ⭐"
echo "   • t_n, t_{n-1}: правильная обработка batch dimension"
echo ""
echo "📋 Что будет происходить:"
echo "   1. Загрузка Teacher модели (на CPU)"
echo "   2. Создание Student модели (на GPU)"
echo "   3. Обучение на 20 итерациях"
echo "   4. Сохранение весов: student_test_20iters_true_cd.pt"
echo "   5. Генерация 3 тестовых изображений"
echo "   6. График лосса: test_cd_loss_curve.png"
echo ""
echo "⏱️  Примерное время: 5-10 минут"
echo "========================================================================"
echo ""

# Проверка CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ CUDA недоступна!"
    exit 1
fi

echo "✅ CUDA доступна: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Проверка данных
if [ ! -d "datadir/latents_good" ] || [ ! -d "datadir/prompts_good" ]; then
    echo "❌ Данные не найдены в datadir/latents_good и datadir/prompts_good"
    echo "   Сначала запустите: bash prepare_full_dataset.sh"
    exit 1
fi

echo "✅ Данные найдены"
echo ""

# Запуск
echo "🚀 Запуск обучения..."
echo ""

python3 test_true_cd_20_iters.py

# Проверка результата
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО!"
    echo "========================================================================"
    echo ""
    echo "📊 Результаты:"
    echo "   • Веса: student_test_20iters_true_cd.pt"
    echo "   • График: test_cd_loss_curve.png"
    echo "   • Изображения: test_cd_outputs/*.png"
    echo ""
    echo "🎯 Следующий шаг:"
    echo "   python3 train_true_consistency_distillation.py  # Полное обучение"
    echo ""
else
    echo ""
    echo "❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТА"
    echo ""
fi

