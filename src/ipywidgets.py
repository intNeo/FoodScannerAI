!pip install ipywidgets -q
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import json
import os
from datetime import datetime
# ---------- 1. ЗАГРУЗКА МОДЕЛИ ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")
model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 101)
model.load_state_dict(torch.load('/content/drive/MyDrive/efficientnet_food101_best.pth', map_location=device))
model = model.to(device)
model.eval()
# ---------- 2. ЗАГРУЗКА НАЗВАНИЙ КЛАССОВ ----------
temp_data = datasets.Food101(root='./data', split='test', transform=None, download=False)
class_names = temp_data.classes
print(f"Загружено {len(class_names)} классов.")
# ---------- 3. ТРАНСФОРМАЦИИ ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ---------- 4. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ----------
def predict_image(image):
    """Принимает PIL Image, возвращает топ-3 предсказания"""
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    top_probs, top_indices = torch.topk(probs, 3)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    results = [(class_names[idx], float(prob)) for idx, prob in zip(top_indices, top_probs)]
    return results
# ---------- 5. ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ----------
def show_prediction(image_path):
    """Загружает изображение, предсказывает и показывает результат"""
    try:
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        # Получаем предсказания
        predictions = predict_image(image)
        # Отображаем изображение
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Левая часть: изображение
        axes[0].imshow(image)
        axes[0].set_title("Загруженное изображение", fontsize=12)
        axes[0].axis('off')
        # Правая часть: результаты
        axes[1].axis('off')
        text = "🏷️ ТОП-3 ПРЕДСКАЗАНИЯ:\n\n"
        for i, (name, prob) in enumerate(predictions, 1):
            text += f"{i}. {name}\n   {prob:.2f}%\n\n"
        # Добавляем статистику
        history_file = '/content/drive/MyDrive/food_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            text += f"📊 Всего запросов: {len(history)}"
        else:
            text += "📊 История пока пуста"
        axes[1].text(0.1, 0.8, text, fontsize=12, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        plt.show()
        # Сохраняем в историю
        save_to_history(image_path, predictions)
    except Exception as e:
        print(f"Ошибка: {e}")
# ---------- 6. ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ ИСТОРИИ ----------
history_file = '/content/drive/MyDrive/food_history.json'
def save_to_history(image_path, predictions):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": os.path.basename(image_path),
        "top1": predictions[0][0],
        "top1_conf": predictions[0][1],
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
    print(f"✅ История сохранена. Всего записей: {len(history)}")
# ---------- 7. СОЗДАНИЕ ИНТЕРФЕЙСА ----------
print("\n" + "="*50)
print("🍔 ПРИЛОЖЕНИЕ ДЛЯ РАСПОЗНАВАНИЯ БЛЮД")
print("="*50)
# Создаём виджеты
upload_button = widgets.FileUpload(
    accept='image/*',  # Только изображения
    multiple=False,
    description='📷 Выберите фото',
    button_style='primary'
)
output = widgets.Output()
# Функция обработки загрузки
def on_upload_change(change):
    with output:
        clear_output(wait=True)
        if upload_button.value:
            # Получаем загруженный файл
            uploaded = list(upload_button.value.values())[0]
            content = uploaded['content']
            # Сохраняем временный файл
            temp_path = '/tmp/temp_image.jpg'
            with open(temp_path, 'wb') as f:
                f.write(content)
            # Показываем результат
            show_prediction(temp_path)
# Привязываем обработчик
upload_button.observe(on_upload_change, names='value')
# Кнопка для просмотра статистики
stats_button = widgets.Button(
    description='📊 Показать статистику',
    button_style='info'
)
stats_output = widgets.Output()
def show_stats(b):
    with stats_output:
        clear_output()
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            total = len(history)
            if total > 0:
                avg_conf = sum(h['top1_conf'] for h in history) / total
                print(f"📊 ВСЕГО ЗАПРОСОВ: {total}")
                print(f"📈 СРЕДНЯЯ УВЕРЕННОСТЬ (TOP-1): {avg_conf:.1f}%")
                print("\n🏆 ЧАСТЫЕ БЛЮДА:")
                from collections import Counter
                top_dishes = Counter(h['top1'] for h in history).most_common(5)
                for dish, count in top_dishes:
                    print(f"  {dish}: {count} раз(а)")
            else:
                print("История пуста")
        else:
            print("История пуста")
stats_button.on_click(show_stats)
# Отображаем интерфейс
display(widgets.VBox([
    widgets.HTML("<h2>🍔 Распознавание блюда по фотографии</h2>"),
    widgets.HTML("<p>Загрузите изображение блюда, и модель покажет топ-3 наиболее вероятных категории.</p>"),
    upload_button,
    stats_button,
    output,
    stats_output
]))
print("\n✅ Интерфейс загружен!")
print("Нажмите на кнопку '📷 Выберите фото' и выберите изображение.")
print("После загрузки появится результат распознавания.")
