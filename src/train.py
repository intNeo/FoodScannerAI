import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# Трансформации для тренировочных данных (с аугментацией)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Трансформации для тестовых данных (без аугментации)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Загрузка датасета
full_train = datasets.Food101(
    root='./data',
    split='train',
    transform=train_transform,
    download=True
)
test_data = datasets.Food101(
    root='./data',
    split='test',
    transform=test_transform,
    download=True
)
print(f"Тренировочных: {len(full_train)}")
print(f"Тестовых: {len(test_data)}")


#Разбиение данных
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_data, val_data = random_split(full_train, [train_size, val_size])
print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

#Создание DataLoader
batch_size = 64
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
print("DataLoader готовы!")

#Проверка загрузки датасета
import matplotlib.pyplot as plt
import numpy as np
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
# Получаем один батч
images, labels = next(iter(train_loader))
# Показываем 8 изображений
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i//4, i%4]
    img = images[i]
    ax.imshow(img.numpy().transpose((1, 2, 0)))
    ax.set_title(f'Класс: {labels[i].item()}')
    ax.axis('off')
plt.tight_layout()
plt.show()

#Вывод изображений с применением обратной нормализации
import matplotlib.pyplot as plt
import numpy as np
# Названия классов
class_names = full_train.classes
def show_normal_image(img, ax=None, title=None):
  # Обратная нормализация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    if ax is None:
        ax = plt.gca()
    ax.imshow(img)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis('off')
# Берём один батч
images, labels = next(iter(train_loader))
# Создаём сетку 2x4
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    row, col = i // 4, i % 4
    show_normal_image(images[i], ax=axes[row, col], title=class_names[labels[i].item()])
plt.tight_layout()
plt.show()

#Запуск оценки архитектуры ResNet-50
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import time
# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Обучение на: {device}")
# 1. Загрузка модели ResNet-50 с предобученными весами
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 101)  # 101 класс Food-101
model = model.to(device)
# 2. Функции потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
# 3. Функции обучения и валидации
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy
# 4. Цикл обучения
epochs = 10
best_val_acc = 0.0
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/content/drive/MyDrive/resnet50_food101_best.pth')
        print(f"✅ Новая лучшая модель сохранена с точностью {best_val_acc:.2f}%")
print("\n Обучение ResNet-50 завершено!")
print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")

#Запуск оценки архитектуры EfficientNet-B0
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Обучение на: {device}")
# 1. Загрузка модели EfficientNet-B0 с предобученными весами
model_eff = models.efficientnet_b0(pretrained=True)
num_features = model_eff.classifier[1].in_features
model_eff.classifier[1] = nn.Linear(num_features, 101)  # 101 класс Food-101
model_eff = model_eff.to(device)
# 2. Функции потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_eff.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
# 3. Цикл обучения
epochs = 10
best_val_acc = 0.0
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} (EfficientNet-B0)")
    # Обучение
    model_eff.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_eff(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
    train_loss = train_loss / train_total
    train_acc = 100. * train_correct / train_total
    # Валидация
    model_eff.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model_eff(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_eff.state_dict(), '/content/drive/MyDrive/efficientnet_food101_best.pth')
        print(f"✅ Новая лучшая модель сохранена с точностью {best_val_acc:.2f}%")
print("\n🎉 Обучение EfficientNet-B0 завершено!")
print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")


#Запуск оценки архитектуры MobileNetV2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Обучение на: {device}")
# 1. Загрузка модели MobileNetV2 с предобученными весами
model_mobilenet = models.mobilenet_v2(pretrained=True)
num_features = model_mobilenet.classifier[1].in_features
model_mobilenet.classifier[1] = nn.Linear(num_features, 101)  # 101 класс Food-101
model_mobilenet = model_mobilenet.to(device)
# 2. Функции потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mobilenet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
# 3. Цикл обучения
epochs = 10
best_val_acc = 0.0
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} (MobileNetV2)")
    # Обучение
    model_mobilenet.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_mobilenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
    train_loss = train_loss / train_total
    train_acc = 100. * train_correct / train_total
    # Валидация
    model_mobilenet.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model_mobilenet(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_mobilenet.state_dict(), '/content/drive/MyDrive/mobilenetv2_food101_best.pth')
        print(f"✅ Новая лучшая модель сохранена с точностью {best_val_acc:.2f}%")
print("\n🎉 Обучение MobileNetV2 завершено!")
print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")


#Запуск оценки архитектур ViT и ConvNeXt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
# 1. Трансформации
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 2. Загружаем тестовые данные 
test_data_full = datasets.Food101(root='./data', split='test', transform=test_transform, download=False)
# 3. Берём первые 1000 изображений
subset_indices = list(range(1000))  # 1000 изображений
test_data_small = Subset(test_data_full, subset_indices)
test_loader_small = DataLoader(test_data_small, batch_size=64, shuffle=False, num_workers=2)
print(f"✅ Тестовая выборка уменьшена до {len(test_data_small)} изображений, батчей: {len(test_loader_small)}")
# 4. Загрузка моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")
print("Загрузка ViT...")
model_vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=101)
model_vit = model_vit.to(device)
model_vit.eval()
print("Загрузка ConvNeXt-Tiny...")
model_convnext = models.convnext_tiny(pretrained=True)
num_features = model_convnext.classifier[2].in_features
model_convnext.classifier[2] = nn.Linear(num_features, 101)
model_convnext = model_convnext.to(device)
model_convnext.eval()
# 5. Функция оценки
def evaluate(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100. * correct / total
# 6. Запуск оценки
print("\nОценка ViT на 1000 изображениях...")
vit_acc = evaluate(model_vit, test_loader_small, device)
print(f"ViT точность (на 1000 тестовых): {vit_acc:.2f}%")
print("\nОценка ConvNeXt на 1000 изображениях...")
convnext_acc = evaluate(model_convnext, test_loader_small, device)
print(f"ConvNeXt точность (на 1000 тестовых): {convnext_acc:.2f}%")
print("\nОценка завершена!")






