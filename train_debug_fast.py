import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

def consistency_distillation_step_debug(latents, text_embeddings, teacher_model, student_model):
    """Один шаг Consistency Distillation с отладкой"""
    print(f"🔍 DEBUG: latents.shape = {latents.shape}, dtype = {latents.dtype}")
    print(f"🔍 DEBUG: text_embeddings.shape = {text_embeddings.shape}, dtype = {text_embeddings.dtype}")
    
    batch_size = latents.shape[0]
    
    # Сэмплируем случайный уровень шума из EDM распределения
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=latents.device)
    sigma = (rnd_normal * teacher_model.edm_config.P_std + teacher_model.edm_config.P_mean).exp()
    
    print(f"🔍 DEBUG: sigma.shape = {sigma.shape}, min = {sigma.min():.6f}, max = {sigma.max():.6f}")
    
    # Добавляем шум к чистым латентам
    noise = torch.randn_like(latents) * sigma
    noisy_latents = latents + noise
    
    print(f"🔍 DEBUG: noisy_latents.shape = {noisy_latents.shape}, min = {noisy_latents.min():.6f}, max = {noisy_latents.max():.6f}")
    
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
    
    print(f"🔍 DEBUG: teacher_denoised.shape = {teacher_denoised.shape}, min = {teacher_denoised.min():.6f}, max = {teacher_denoised.max():.6f}")
    
    # Student: также делает один шаг денойзинга
    student_output = student_model.model_forward_wrapper(
        noisy_latents,
        sigma,
        text_embeddings,
        student_model.dit,
        mask_ratio=0.0
    )
    student_denoised = student_output['sample']
    
    print(f"🔍 DEBUG: student_denoised.shape = {student_denoised.shape}, min = {student_denoised.min():.6f}, max = {student_denoised.max():.6f}")
    
    # Consistency loss
    loss = nn.MSELoss()(student_denoised, teacher_denoised)
    
    print(f"🔍 DEBUG: loss = {loss.item():.6f}")
    
    return loss

# =======================
# Запуск отладки
# =======================
if __name__ == "__main__":
    print("🔍 ОТЛАДКА с быстрой загрузкой моделей")
    
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
        exit(1)

    # =======================
    # Загрузка данных (только один батч)
    # =======================
    print("Загрузка данных для отладки...")
    latents_dir = r"C:\newTry2\train\datadir\latents_good"
    prompts_dir = r"C:\newTry2\train\datadir\prompts_good"
    
    dataset = LatentPromptDataset(latents_dir, prompts_dir)
    
    # Берем только первый батч для отладки
    latents, prompts = dataset[0]
    latents = latents.unsqueeze(0)  # Добавляем batch dimension
    
    print(f"✅ Загружен один батч: latents.shape = {latents.shape}")
    print(f"✅ Промпт: {prompts[:50]}...")
    
    # =======================
    # Отладочный шаг
    # =======================
    print("\n🔍 Выполняем отладочный шаг...")
    try:
        text_embeddings = get_text_embeddings([prompts], teacher_model)
        loss = consistency_distillation_step_debug(latents, text_embeddings, teacher_model, student_model)
        
        print(f"\n✅ Отладка завершена успешно!")
        print(f"📊 Loss: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("⚠️ ВНИМАНИЕ: Loss содержит NaN!")
        else:
            print("✅ Loss корректный (не NaN)")
            
    except Exception as e:
        print(f"❌ Ошибка при отладке: {e}")
        import traceback
        traceback.print_exc()

