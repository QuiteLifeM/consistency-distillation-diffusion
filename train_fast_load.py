import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Настройка для CPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =======================
# Dataset для предвычисленных латентов
# =======================
class LatentPromptDataset(Dataset):
    """Dataset для загрузки предвычисленных латентов и промптов"""
    def __init__(self, latents_dir, prompts_dir, preload=True):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        self.preload = preload
        
        # Получаем отсортированные списки файлов
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        assert len(self.latent_files) == len(self.prompt_files), \
            f"Количество латентов ({len(self.latent_files)}) != количество промптов ({len(self.prompt_files)})"
        
        # Предзагрузка данных для ускорения
        if preload:
            print("⚡ Предзагрузка данных в память...")
            self.latents_cache = []
            self.prompts_cache = []
            
            for i in tqdm(range(len(self.latent_files)), desc="Загрузка латентов"):
                latent_path = os.path.join(self.latents_dir, self.latent_files[i])
                latent = torch.load(latent_path)
                self.latents_cache.append(latent)
                
                prompt_path = os.path.join(self.prompts_dir, self.prompt_files[i])
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                self.prompts_cache.append(prompt)
            
            print(f"✅ Предзагружено {len(self.latents_cache)} латентов и промптов")
        else:
            print(f"✅ Загружено {len(self.latent_files)} латентов и промптов (ленивая загрузка)")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        if self.preload:
            # Используем предзагруженные данные
            return self.latents_cache[idx], self.prompts_cache[idx]
        else:
            # Ленивая загрузка
            latent_path = os.path.join(self.latents_dir, self.latent_files[idx])
            latent = torch.load(latent_path)
            
            prompt_path = os.path.join(self.prompts_dir, self.prompt_files[idx])
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            return latent, prompt

# =======================
# Функции для Consistency Distillation
# =======================
def get_text_embeddings(prompts, model):
    """Получение текстовых эмбеддингов через модель"""
    tokenized = model.tokenizer.tokenize(prompts)
    device = next(model.parameters()).device
    input_ids = tokenized['input_ids'].to(device)
    
    with torch.no_grad():
        text_embeddings = model.text_encoder.encode(input_ids)[0]
    
    return text_embeddings

def consistency_distillation_step(latents, text_embeddings, teacher_model, student_model):
    """Один шаг Consistency Distillation"""
    batch_size = latents.shape[0]
    
    # Сэмплируем случайный уровень шума из EDM распределения
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=latents.device)
    sigma = (rnd_normal * teacher_model.edm_config.P_std + teacher_model.edm_config.P_mean).exp()
    
    # Добавляем шум к чистым латентам
    noise = torch.randn_like(latents) * sigma
    noisy_latents = latents + noise
    
    # Teacher: делает один шаг денойзинга
    with torch.no_grad():
        teacher_output = teacher_model.model_forward_wrapper(
            noisy_latents,
            sigma,
            text_embeddings,
            teacher_model.dit,
            mask_ratio=0.0
        )
        teacher_denoised = teacher_output['sample']
    
    # Student: также делает один шаг денойзинга
    student_output = student_model.model_forward_wrapper(
        noisy_latents,
        sigma,
        text_embeddings,
        student_model.dit,
        mask_ratio=0.0
    )
    student_denoised = student_output['sample']
    
    # Consistency loss
    loss = nn.MSELoss()(student_denoised, teacher_denoised)
    
    return loss

# =======================
# Тренировочный цикл
# =======================
def custom_collate(batch):
    """Кастомная функция для обработки батчей"""
    latents_list = []
    prompts_list = []
    
    for latent, prompt in batch:
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        
        latents_list.append(latent)
        prompts_list.append(prompt)
    
    latents_batch = torch.stack(latents_list, dim=0)
    return latents_batch, prompts_list

def train_consistency_distillation(dataloader, teacher_model, student_model, optimizer, num_epochs=5):
    """Обучение студента через Consistency Distillation"""
    losses_history = []
    
    print(f"\n{'='*60}")
    print(f"Начало обучения: {num_epochs} эпох, {len(dataloader)} батчей на эпоху")
    print(f"🚀 БЫСТРАЯ ЗАГРУЗКА: batch_size=1, num_workers=0, float32")
    print(f"⏱️  Примерное время: ~{len(dataloader) * 1.2 / 3600:.1f} часов на эпоху")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        student_model.train()
        num_batches = 0
        
        for i, (latents, prompts) in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch+1}/{num_epochs}", unit="батч")):
            try:
                latents = latents.float()
                
                if latents.dim() == 3:
                    latents = latents.unsqueeze(0)
                
                assert latents.shape[1] == 4, f"Ожидается 4 канала, получено {latents.shape[1]}"
                assert latents.shape[2] == 64 and latents.shape[3] == 64, \
                    f"Ожидается размер 64x64, получено {latents.shape[2]}x{latents.shape[3]}"
                
                text_embeddings = get_text_embeddings(prompts, teacher_model)
                loss = consistency_distillation_step(latents, text_embeddings, teacher_model, student_model)
                
                if torch.isnan(loss):
                    print(f"⚠️ NaN loss в батче {i}, пропускаем...")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.dit.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Обновляем описание прогресс-бара
                if i % 10 == 0:
                    tqdm.write(f"Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"⚠️ Ошибка в батче {i}: {e}")
                continue
        
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            losses_history.append(avg_epoch_loss)
            
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch+1} завершена | Средний Loss: {avg_epoch_loss:.6f}")
            print(f"{'='*60}\n")
            
            checkpoint_path = f"student_checkpoint_epoch_{epoch+1}.pt"
            torch.save(student_model.dit.state_dict(), checkpoint_path)
            print(f"💾 Сохранен чекпоинт: {checkpoint_path}\n")
        else:
            print(f"⚠️ Эпоха {epoch+1}: Нет успешных батчей!")
    
    print("✅ Обучение завершено!")
    return losses_history

# =======================
# Запуск обучения
# =======================
if __name__ == "__main__":
    print("🚀 БЫСТРАЯ ЗАГРУЗКА Consistency Distillation")
    print("⚡ Модели загружаются за секунды!")
    print("🛡️ СТАБИЛЬНАЯ ВЕРСИЯ: batch_size=1 для Windows")
    
    # Проверяем наличие сохраненных моделей
    if not os.path.exists("teacher_model_ready.pt") or not os.path.exists("student_model_ready.pt"):
        print("❌ Сохраненные модели не найдены!")
        print("Сначала запустите: python save_models.py")
        exit(1)
    
    # =======================
    # БЫСТРАЯ ЗАГРУЗКА МОДЕЛЕЙ
    # =======================
    print("⚡ Быстрая загрузка моделей...")
    try:
        teacher_model = torch.load("teacher_model_ready.pt", map_location="cpu")
        print("✅ Учитель загружен за секунды!")
        
        student_model = torch.load("student_model_ready.pt", map_location="cpu")
        print("✅ Студент загружен за секунды!")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке моделей: {e}")
        print("Попробуйте пересоздать модели: python save_models.py")
        exit(1)

    # Оптимизатор для студента
    optimizer = optim.Adam(student_model.dit.parameters(), lr=1e-5)
    print(f"✅ Оптимизатор настроен с lr={1e-5}\n")
    
    # =======================
    # Загрузка данных
    # =======================
    print("Загрузка данных...")
    latents_dir = r"C:\newTry2\train\datadir\latents_good"
    prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    # Выбираем режим загрузки
    print("⚡ Режимы загрузки:")
    print("1. Предзагрузка (быстро, но много RAM)")
    print("2. Ленивая загрузка (медленно, но мало RAM)")
    
    # По умолчанию предзагрузка для скорости
    preload_mode = True
    print(f"✅ Выбран режим: {'Предзагрузка' if preload_mode else 'Ленивая загрузка'}")
    
    dataset = LatentPromptDataset(latents_dir, prompts_dir, preload=preload_mode)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Вернули к 1 для стабильности
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate
    )
    print(f"✅ DataLoader создан: {len(dataset)} сэмплов, batch_size=1\n")
    
    # =======================
    # Запуск обучения
    # =======================
    num_epochs = 5
    print(f"🚀 Начинаем обучение на {num_epochs} эпох...")
    print(f"⏱️  Примерное время: ~{len(dataloader) * 1.2 * num_epochs / 3600:.1f} часов")
    print(f"📊 Прогресс-бар покажет детальную информацию о каждом батче")
    
    losses = train_consistency_distillation(
        dataloader, teacher_model, student_model, optimizer, num_epochs
    )
    
    # Сохраняем финальную модель
    final_model_path = "student_final_fast_load.pt"
    torch.save(student_model.dit.state_dict(), final_model_path)
    print(f"\n💾 Финальная модель сохранена: {final_model_path}")
    
    # График лосса
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linewidth=2)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Consistency Distillation Training Loss (Fast Load)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss_fast_load.png', dpi=150, bbox_inches='tight')
    print(f"📊 График сохранен: training_loss_fast_load.png")
