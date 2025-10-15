import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from micro_diffusion.micro_diffusion.models.model import create_latent_diffusion

# Настройка для CPU
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
# Функции для Consistency Distillation на CPU (отладочная версия)
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
    
    return text_embeddings  # Оставляем на том же устройстве (CPU)

def consistency_distillation_step_debug(latents, text_embeddings, teacher_model, student_model):
    """
    Один шаг Consistency Distillation на CPU (отладочная версия)
    """
    print(f"🔍 DEBUG: latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"🔍 DEBUG: latents min: {latents.min():.6f}, max: {latents.max():.6f}")
    print(f"🔍 DEBUG: latents has NaN: {torch.isnan(latents).any()}")
    print(f"🔍 DEBUG: text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")
    print(f"🔍 DEBUG: text_embeddings min: {text_embeddings.min():.6f}, max: {text_embeddings.max():.6f}")
    print(f"🔍 DEBUG: text_embeddings has NaN: {torch.isnan(text_embeddings).any()}")
    
    batch_size = latents.shape[0]
    
    # Сэмплируем случайный уровень шума из EDM распределения
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=latents.device)
    sigma = (rnd_normal * teacher_model.edm_config.P_std + teacher_model.edm_config.P_mean).exp()
    
    print(f"🔍 DEBUG: sigma shape: {sigma.shape}, dtype: {sigma.dtype}, min: {sigma.min():.6f}, max: {sigma.max():.6f}")
    
    # Добавляем шум к чистым латентам
    noise = torch.randn_like(latents) * sigma
    noisy_latents = latents + noise
    
    print(f"🔍 DEBUG: noisy_latents shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}")
    print(f"🔍 DEBUG: noisy_latents min: {noisy_latents.min():.6f}, max: {noisy_latents.max():.6f}")
    
    # Teacher: делает один шаг денойзинга с текущего уровня шума
    with torch.no_grad():
        print("🔍 DEBUG: Запускаем teacher...")
        teacher_output = teacher_model.model_forward_wrapper(
            noisy_latents,
            sigma,
            text_embeddings,
            teacher_model.dit,
            mask_ratio=0.0
        )
        teacher_denoised = teacher_output['sample']
        print(f"🔍 DEBUG: teacher_denoised shape: {teacher_denoised.shape}, dtype: {teacher_denoised.dtype}")
        print(f"🔍 DEBUG: teacher_denoised min: {teacher_denoised.min():.6f}, max: {teacher_denoised.max():.6f}")
    
    # Student: также делает один шаг денойзинга
    print("🔍 DEBUG: Запускаем student...")
    student_output = student_model.model_forward_wrapper(
        noisy_latents,
        sigma,
        text_embeddings,
        student_model.dit,
        mask_ratio=0.0
    )
    student_denoised = student_output['sample']
    print(f"🔍 DEBUG: student_denoised shape: {student_denoised.shape}, dtype: {student_denoised.dtype}")
    print(f"🔍 DEBUG: student_denoised min: {student_denoised.min():.6f}, max: {student_denoised.max():.6f}")
    
    # Consistency loss: студент должен предсказывать то же, что и учитель
    loss = nn.MSELoss()(student_denoised, teacher_denoised)
    print(f"🔍 DEBUG: loss: {loss.item():.6f}")
    
    return loss

# =======================
# Тренировочный цикл (отладочная версия)
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

# =======================
# Запуск обучения
# =======================
if __name__ == "__main__":
    print("🔍 ОТЛАДОЧНАЯ ВЕРСИЯ - только 1 батч для диагностики")
    
    # Пути к данным
    latents_dir = r"C:\newTry2\train\datadir\latents_good"
    prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    # =======================
    # Загрузка моделей учителя и студента на CPU
    # =======================
    print("Загрузка модели учителя на CPU...")
    try:
        teacher_model = create_latent_diffusion(
            latent_res=64,
            in_channels=4,
            pos_interp_scale=2.0,
            precomputed_latents=False,
            dtype="float32"
        )
        print("✅ Модель создана, оставляем на CPU...")
        teacher_model = teacher_model.to("cpu")
        print("✅ Модель оставлена на CPU")
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        exit(1)

    # Загружаем веса учителя
    print("Загружаем веса учителя...")
    try:
        teacher_weights = torch.load("./micro_diffusion/trained_models/teacher.pt", map_location="cpu")
        teacher_model.dit.load_state_dict(teacher_weights, strict=False)
        teacher_model.eval()
        print("✅ Модель учителя загружена на CPU")
    except Exception as e:
        print(f"❌ Ошибка при загрузке весов: {e}")
        exit(1)

    print("\nСоздание модели студента на CPU...")
    student_model = create_latent_diffusion(
        latent_res=64,
        in_channels=4,
        pos_interp_scale=2.0,
        precomputed_latents=False,
        dtype="float32"
    ).to("cpu")

    # Инициализируем студента весами учителя
    student_model.dit.load_state_dict(teacher_weights, strict=False)
    student_model.train()
    print("✅ Модель студента создана и инициализирована весами учителя на CPU")

    # Создаем датасет и даталоадер (только 1 батч для отладки)
    print("Загрузка данных...")
    dataset = LatentPromptDataset(latents_dir, prompts_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=0,  # Отключаем multiprocessing для отладки
        collate_fn=custom_collate
    )
    print(f"✅ DataLoader создан: {len(dataset)} сэмплов, batch_size=1\n")
    
    # Тестируем только первый батч
    print("🔍 Тестируем первый батч...")
    for i, (latents, prompts) in enumerate(dataloader):
        if i >= 1:  # Только первый батч
            break
            
        print(f"\n{'='*60}")
        print(f"Тестируем батч {i}")
        print(f"{'='*60}")
        
        try:
            # Оставляем латенты на CPU
            latents = latents.float()
            
            # Проверяем размерность латентов
            if latents.dim() == 3:
                latents = latents.unsqueeze(0)
            
            # Проверяем, что латенты имеют правильный размер
            assert latents.shape[1] == 4, f"Ожидается 4 канала, получено {latents.shape[1]}"
            assert latents.shape[2] == 64 and latents.shape[3] == 64, \
                f"Ожидается размер 64x64, получено {latents.shape[2]}x{latents.shape[3]}"
            
            # Проверяем на NaN в исходных данных
            if torch.isnan(latents).any():
                print(f"❌ ИСХОДНЫЕ ЛАТЕНТЫ СОДЕРЖАТ NaN!")
                print(f"   latents min: {latents.min():.6f}, max: {latents.max():.6f}")
                print(f"   latents has NaN: {torch.isnan(latents).any()}")
                continue
            
            # Получаем текстовые эмбеддинги
            text_embeddings = get_text_embeddings(prompts, teacher_model)
            
            # Consistency distillation step на CPU
            loss = consistency_distillation_step_debug(
                latents, text_embeddings, teacher_model, student_model
            )
            
            print(f"✅ Успешно! Loss: {loss.item():.6f}")
            
        except Exception as e:
            print(f"❌ Ошибка в батче {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🔍 Отладка завершена!")
