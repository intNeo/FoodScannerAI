# Анализ ошибок лучшей модели EfficientNet-B0 на тестовой выборке Food-101.
# Загружает модель, вычисляет матрицу ошибок, метрики и показывает примеры.


import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
import random
import os

# ==================== НАСТРОЙКИ ====================
# Путь к файлу с весами модели
MODEL_WEIGHTS_PATH = "models/efficientnet_b0_food101.pth"
# Папка, куда будут скачаны данные Food-101 
DATA_ROOT = "./data"
# Количество изображений для анализа 
SUBSET_SIZE = 1000  # 1000 изображений из тестовой выборки

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

# Проверяем, существует ли файл весов
if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Файл весов не найден: {MODEL_WEIGHTS_PATH}\n"
        "Укажите правильный путь или скачайте веса с Google Диска."
    )

model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 101)  # 101 класс Food-101
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Модель загружена.")

# ==================== ЗАГРУЗКА ДАННЫХ ====================
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Загружаем тестовые данные 
test_data_full = datasets.Food101(root=DATA_ROOT, split='test',
                                  transform=test_transform, download=True)

# Берём только SUBSET_SIZE изображений для ускорения
subset_indices = list(range(min(SUBSET_SIZE, len(test_data_full))))
test_data = Subset(test_data_full, subset_indices)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=True)

print(f"Тестовая выборка: {len(test_data)} изображений.")

# ==================== ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ ====================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Тестирование'):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

class_names = test_data_full.classes

# ==================== МАТРИЦА ОШИБОК ====================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(16, 12))
sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[:20], yticklabels=class_names[:20])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (первые 20 классов)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_20.png')   # Сохраняем в текущую папку
plt.show()

# ==================== КЛАССЫ С НАИБОЛЬШИМ ЧИСЛОМ ОШИБОК ====================
errors_per_class = cm.sum(axis=1) - np.diag(cm)
most_mistaken_classes = np.argsort(errors_per_class)[-5:][::-1]

print("\n=== Классы с наибольшим числом ошибок (пропусков) ===")
for idx in most_mistaken_classes:
    total = cm[idx].sum()
    if total > 0:
        print(f"  {class_names[idx]} — {errors_per_class[idx]} ошибок из {total}")

# ==================== МЕТРИКИ ====================
unique_labels = sorted(set(all_labels))
precision = precision_score(all_labels, all_preds, average='macro',
                            labels=unique_labels, zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro',
                      labels=unique_labels, zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro',
              labels=unique_labels, zero_division=0)
accuracy = accuracy_score(all_labels, all_preds)

print("\n=== Метрики (macro avg) ===")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-score:  {f1:.4f}")

# ==================== ПРИМЕРЫ УСПЕШНЫХ И ОШИБОЧНЫХ ПРЕДСКАЗАНИЙ ====================
wrong_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_preds))
                 if true != pred]
correct_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_preds))
                   if true == pred]

random.seed(42)
sample_wrong = random.sample(wrong_indices, min(3, len(wrong_indices)))
sample_correct = random.sample(correct_indices, min(3, len(correct_indices)))

def imshow(img_tensor, ax, title=''):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# Успешные предсказания
if sample_correct:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, idx in enumerate(sample_correct):
        img_tensor = test_data[idx][0]
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        title = f"Истинный: {class_names[true_label]}\nПредсказанный: {class_names[pred_label]}"
        imshow(img_tensor, axes[i], title)
    plt.suptitle("Успешные предсказания", fontsize=14)
    plt.tight_layout()
    plt.savefig('correct_predictions.png')
    plt.show()

# Ошибочные предсказания
if sample_wrong:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, idx in enumerate(sample_wrong):
        img_tensor = test_data[idx][0]
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        title = f"Истинный: {class_names[true_label]}\nПредсказанный: {class_names[pred_label]}"
        imshow(img_tensor, axes[i], title)
    plt.suptitle("Ошибочные предсказания", fontsize=14)
    plt.tight_layout()
    plt.savefig('wrong_predictions.png')
    plt.show()

print("\n✅ Анализ ошибок завершён!")