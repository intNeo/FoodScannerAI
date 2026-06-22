import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------- 1. Загрузка модели --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Путь к весам 
# скачать веса отдельно и положить в папку models/
MODEL_PATH = 'models/efficientnet_b0_food101.pth'

def load_model(weights_path=MODEL_PATH):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 101)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Загружаем модель при импорте (для быстрого доступа)
model = load_model()

# -------------------- 2. Трансформации --------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 3. Функция предсказания --------------------
def predict(image):
    """
    Принимает PIL Image или путь к файлу.
    Возвращает список из трёх кортежей (класс, вероятность_в_%).
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Подайте PIL Image или путь к файлу")

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    top_probs, top_indices = torch.topk(probs, 3)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Названия классов 
    class_names = load_class_names()  # см. функцию ниже

    results = [(class_names[idx], float(prob)) for idx, prob in zip(top_indices, top_probs)]
    return results

# -------------------- 4. Загрузка названий классов --------------------
def load_class_names():
    from torchvision import datasets
    temp = datasets.Food101(root='./data', split='test', download=False)
    return temp.classes

# -------------------- 5. Пример использования --------------------
if __name__ == "__main__":
    # Пример: предсказание для файла
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        preds = predict(img_path)
        print("Топ-3 предсказания:")
        for i, (name, prob) in enumerate(preds, 1):
            print(f"{i}. {name} – {prob:.2f}%")
    else:
        print("Укажите путь к изображению: python inference.py image.jpg")