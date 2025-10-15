import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion

# Настройка CUDA для экономии памяти
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =======================
# Dataset для предвычисленных латентов
# =======================
class LatentPromptDataset(Dataset):
    """Dataset для загрузки предвычисленных латентов и промптов"""
    def __init__(self, latents_dir, prompts_dir):
        self.latents_dir = latents_dir
        self.prompts_dir = prompts_dir
        
        # Получаем отсортированные списки файлов
        self.latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith('.pt')])
        self.prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
        
        assert len(self.latent_files) == len(self.prompt_files), \
            f"Количество латентов ({len(self.latent_files)}) != количество промптов ({len(self.prompt_files)})"
        
        print(f"✅ Загружено {len(self.latent_files)} латентов и промптов")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # Загружаем латент
        latent_path = os.path.join(self.latents_dir, self.latent_files[idx])
        latent = torch.load(latent_path)
        
        # Загружаем промпт
        prompt_path = os.path.join(self.prompts_dir, self.prompt_files[idx])
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        return latent, prompt

# =======================
# Функции для Consistency Distillation
# =======================
def get_text_embeddings(prompts, model):
    """Получение текстовых эмбеддингов через модель"""
    # Используем встроенный токенизатор и текстовый энкодер модели
    tokenized = model.tokenizer.tokenize(prompts)
    device = next(model.parameters()).device
    input_ids = tokenized['input_ids'].to(device)
    
    with torch.no_grad():
        # Получаем эмбеддинги через text_encoder модели
        text_embeddings = model.text_encoder.encode(input_ids)[0]
    
    return text_embeddings.to("cuda")  # Перемещаем на GPU для обучения

def consistency_distillation_step(latents, text_embeddings, teacher_model, student_model):
    """
    Один шаг Consistency Distillation
    """
    batch_size = latents.shape[0]
    
    # Сэмплируем случайный уровень шума из EDM распределения
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=latents.device)
    sigma = (rnd_normal * teacher_model.edm_config.P_std + teacher_model.edm_config.P_mean).exp()
    
    # Добавляем шум к чистым латентам
    noise = torch.randn_like(latents) * sigma
    noisy_latents = latents + noise
    
    # Teacher: делает один шаг денойзинга с текущего уровня шума
    with torch.no_grad():
        # Перемещаем данные на устройство учителя
        teacher_latents = noisy_latents.to(next(teacher_model.parameters()).device)
        teacher_sigma = sigma.to(next(teacher_model.parameters()).device)
        teacher_embeddings = text_embeddings.to(next(teacher_model.parameters()).device)
        
        teacher_output = teacher_model.model_forward_wrapper(
            teacher_latents.float(),
            teacher_sigma,
            teacher_embeddings.float(),
            teacher_model.dit,
            mask_ratio=0.0
        )
        teacher_denoised = teacher_output['sample'].to("cuda")  # Возвращаем на GPU
    
    # Student: также делает один шаг денойзинга
    student_output = student_model.model_forward_wrapper(
        noisy_latents.float(),
        sigma,
        text_embeddings.float(),
        student_model.dit,
        mask_ratio=0.0
    )
    student_denoised = student_output['sample']
    
    # Consistency loss: студент должен предсказывать то же, что и учитель
    loss = nn.MSELoss()(student_denoised, teacher_denoised)
    
    return loss

# =======================
# Тренировочный цикл
# =======================
def custom_collate(batch):
    """
    Кастомная функция для обработки батчей
    Обрабатывает латенты разных размеров и объединяет промпты
    """
    latents_list = []
    prompts_list = []
    
    for latent, prompt in batch:
        # Убеждаемся, что латент имеет правильную размерность [C, H, W]
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # Убираем лишнюю batch dimension
        
        latents_list.append(latent)
        prompts_list.append(prompt)
    
    # Стакаем латенты в батч [B, C, H, W]
    latents_batch = torch.stack(latents_list, dim=0)
    
    return latents_batch, prompts_list

def train_consistency_distillation(dataloader, teacher_model, student_model, optimizer, num_epochs=10):
    """
    Обучение студента через Consistency Distillation
    """
    losses_history = []
    
    print(f"\n{'='*60}")
    print(f"Начало обучения: {num_epochs} эпох, {len(dataloader)} батчей на эпоху")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        student_model.train()
        num_batches = 0
        
        for i, (latents, prompts) in enumerate(dataloader):
            try:
                # Перемещаем латенты на GPU
                latents = latents.to("cuda").float()
                
                # Проверяем размерность латентов
                if latents.dim() == 3:
                    # Если нет batch dim, добавляем
                    latents = latents.unsqueeze(0)
                
                # Проверяем, что латенты имеют правильный размер
                assert latents.shape[1] == 4, f"Ожидается 4 канала, получено {latents.shape[1]}"
                assert latents.shape[2] == 64 and latents.shape[3] == 64, \
                    f"Ожидается размер 64x64, получено {latents.shape[2]}x{latents.shape[3]}"
                
                # Получаем текстовые эмбеддинги
                text_embeddings = get_text_embeddings(prompts, teacher_model)
                
                # Consistency distillation step
                loss = consistency_distillation_step(
                    latents, text_embeddings, teacher_model, student_model
                )
                
                # Обратное распространение
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(student_model.dit.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Логирование
                if i % 10 == 0:
                    print(f"Эпоха [{epoch+1}/{num_epochs}] | Батч [{i}/{len(dataloader)}] | Loss: {loss.item():.6f}")
                    # Очищаем память каждые 10 батчей
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️ Ошибка в батче {i}: {e}")
                # Принудительная очистка памяти при ошибке
                torch.cuda.empty_cache()
                continue
        
        # Средний лосс за эпоху
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            losses_history.append(avg_epoch_loss)
            
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch+1} завершена | Средний Loss: {avg_epoch_loss:.6f}")
            print(f"{'='*60}\n")
            
            # Сохраняем чекпоинт каждые 2 эпохи
            if (epoch + 1) % 2 == 0:
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
    # Очищаем память GPU перед началом
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Принудительная очистка памяти
        torch.cuda.synchronize()
        print(f"🧹 GPU память очищена. Свободно: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Пути к данным
    latents_dir = r"C:\newTry2\train\datadir\latents"
    prompts_dir = r"C:\newTry2\train\datadir\prompts"
    
    # =======================
    # Загрузка моделей учителя и студента
    # =======================
    print("Загрузка модели учителя...")
    try:
        teacher_model = create_latent_diffusion(
            latent_res=64,  # Возвращаем к оригинальному размеру для совместимости с весами
            in_channels=4,  # VAE latents имеют 4 канала
            pos_interp_scale=2.0,  # Возвращаем оригинальный scale
            precomputed_latents=False,  # Мы НЕ используем предвычисленные латенты в forward
            dtype="float16"  # Используем float16 для экономии памяти
        )
        print("✅ Модель создана, оставляем на CPU для экономии памяти...")
        teacher_model = teacher_model.to("cpu")
        print("✅ Модель оставлена на CPU")
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        print("Попробуем загрузить на CPU...")
        teacher_model = create_latent_diffusion(
            latent_res=64,
            in_channels=4,
            pos_interp_scale=2.0,
            precomputed_latents=False,
            dtype="float16"
        ).to("cpu")
        print("✅ Модель загружена на CPU")

    # Загружаем веса учителя
    print("Загружаем веса учителя...")
    try:
        device = "cuda" if next(teacher_model.parameters()).is_cuda else "cpu"
        teacher_weights = torch.load("./micro_diffusion/trained_models/teacher.pt", map_location=device)
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        teacher_model.eval()  # Учитель всегда в режиме eval
        print("✅ Модель учителя загружена")
    except Exception as e:
        print(f"❌ Ошибка при загрузке весов: {e}")
        print("Попробуем загрузить веса на CPU...")
        teacher_weights = torch.load("./micro_diffusion/trained_models/teacher.pt", map_location="cpu")
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        teacher_model.eval()
        print("✅ Веса загружены на CPU")

    print("\nСоздание модели студента...")
    student_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float16"
    ).to("cuda")

    # Инициализируем студента весами учителя
    student_model.dit.load_state_dict(teacher_weights, strict=False)
    student_model.train()  # Студент в режиме тренировки
    print("✅ Модель студента создана и инициализирована весами учителя")

    # Оптимизатор для студента
    optimizer = optim.Adam(student_model.dit.parameters(), lr=1e-5)
    print(f"✅ Оптимизатор настроен с lr={1e-5}\n")
    
    # Создаем датасет и даталоадер
    print("Загрузка данных...")
    dataset = LatentPromptDataset(latents_dir, prompts_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Уменьшили для экономии памяти 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate  # Используем кастомную функцию для батчинга
    )
    print(f"✅ DataLoader создан: {len(dataset)} сэмплов, batch_size=1\n")
    
    # Запускаем обучение
    num_epochs = 10
    losses = train_consistency_distillation(
        dataloader, teacher_model, student_model, optimizer, num_epochs
    )
    
    # Сохраняем финальную модель
    final_model_path = "student_final.pt"
    torch.save(student_model.dit.state_dict(), final_model_path)
    print(f"\n💾 Финальная модель сохранена: {final_model_path}")
    
    # График лосса
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linewidth=2)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Consistency Distillation Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print(f"📊 График сохранен: training_loss.png")