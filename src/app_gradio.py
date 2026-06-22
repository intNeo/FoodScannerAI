!pip install gradio -q
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import json
import os
from datetime import datetime
# ---------- 1. ЗАГРУЗКА МОДЕЛИ ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загружаем лучшую модель EfficientNet-B0
model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 101)
model.load_state_dict(torch.load('/content/drive/MyDrive/efficientnet_food101_best.pth', map_location=device))
model = model.to(device)
model.eval()
# ---------- 2. ПРЕДОБРАБОТКА ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ---------- 3. НАЗВАНИЯ КЛАССОВ ----------
# Загружаем датасет только для получения списка классов (данные уже скачаны)
test_data = datasets.Food101(root='./data', split='test', transform=transform, download=False)
class_names = test_data.classes
# ---------- 4. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ----------
def predict(image):
    if image is None:
        return {}
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    top_probs, top_indices = torch.topk(probs, 3)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    result = {class_names[idx]: float(prob) for idx, prob in zip(top_indices, top_probs)}
    return result
# ---------- 5. ЛОГИРОВАНИЕ ИСТОРИИ ----------
history_file = '/content/drive/MyDrive/food_history.json'
def save_history(image, predictions):
    if not predictions:
        return "⚠️ Нет предсказаний для сохранения."
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top1": list(predictions.keys())[0],
        "top1_conf": list(predictions.values())[0],
        "all_top3": predictions
    }
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    return f"Запрос сохранён! Всего записей: {len(history)}"
def get_stats():
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        total = len(history)
        if total > 0:
            avg_conf = sum(h['top1_conf'] for h in history) / total
        else:
            avg_conf = 0
        # Считаем топ-3 самых частых блюд
        from collections import Counter
        top_dishes = Counter(h['top1'] for h in history).most_common(3)
        top_str = ", ".join([f"{d} ({c})" for d, c in top_dishes])
        return f"📊 Всего запросов: {total}\n📈 Средняя уверенность (top-1): {avg_conf:.1f}%\n🏆 Частые блюда: {top_str}"
    else:
        return "📊 История пока пуста."
# ---------- 6. ИНТЕРФЕЙС ----------
with gr.Blocks(title="Food-101 Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🍔 Распознавание блюда по фотографии")
    gr.Markdown("Загрузите изображение блюда, и модель покажет **3 наиболее вероятных** категории.")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📷 Загрузите фото")
            submit_btn = gr.Button("🔍 Распознать!", variant="primary")
        with gr.Column(scale=1):
            output_labels = gr.Label(label="🏷️ Топ 3 предсказания", num_top_classes=3)
            stats_text = gr.Textbox(label="📈 Статистика", interactive=False)
            history_text = gr.Textbox(label="📜 Последнее сохранение", interactive=False)
    def process(image):
        if image is None:
            return {}, "Сначала загрузите изображение.", "", ""
        preds = predict(image)
        stats = get_stats()
        save_msg = save_history(image, preds)
        return preds, stats, save_msg, save_msg  # возвращаем 4 значения
    submit_btn.click(
        process,
        inputs=input_image,
        outputs=[output_labels, stats_text, history_text, history_text]
    )
    # Кнопка обновления статистики
    refresh_btn = gr.Button("🔄 Обновить статистику")
    refresh_btn.click(get_stats, outputs=stats_text)
# ---------- 7. ЗАПУСК ----------
demo.launch(share=True, debug= True)
