import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from datasets import load_dataset
from PIL import Image

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# 데이터셋 로드
train_dataset = load_dataset("xodhks/EmoSet118K_MonetStyle", split="train")
test_dataset = load_dataset("xodhks/Children_Sketch", split="train")

# 테스트 데이터셋의 유효 라벨 목록
test_valid_label_indices = [0, 1, 4, 5]  # Children_Sketch에 존재하는 라벨 인덱스만 포함

# ResNet 모델 로드 (예: ResNet-50)
model = models.resnet50(pretrained=True).to(device)

# 모델 저장을 위한 디렉토리 생성
os.makedirs("top_models", exist_ok=True)
top_models = []

# 데이터셋 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 전처리 함수
def preprocess_data(batch):
    # 이미지가 이미 PIL 객체인 경우
    if isinstance(batch['image'], list):
        batch['image'] = [img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in batch['image']]
    elif isinstance(batch['image'], Image.Image):
        batch['image'] = batch['image'].convert("RGB")
    else:
        batch['image'] = Image.open(batch['image']).convert("RGB")
    
    # transform 적용
    batch['image'] = [transform(img) for img in batch['image']]
    return batch

train_dataset = train_dataset.with_transform(preprocess_data)
test_dataset = test_dataset.with_transform(preprocess_data)

# DataLoader 설정
def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # 텐서로 변환
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# 평가 함수
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for pred, label in zip(preds, labels):
                if pred.item() in test_valid_label_indices:
                    if pred.item() == label.item():
                        correct += 1
                total += 1

    accuracy = 100 * correct / total
    return accuracy

# 모델 저장 함수
def save_top_models(epoch, accuracy, model, top_models):
    model_filename = f"model_epoch_{epoch + 1}_accuracy_{accuracy:.2f}.pth"
    model_path = os.path.join("top_models", model_filename)
    top_models.append((accuracy, model_path))
    top_models = sorted(top_models, key=lambda x: x[0], reverse=True)[:10]
    torch.save(model.state_dict(), model_path)
    print("\nTop 10 Models (by accuracy):")
    for i, (acc, path) in enumerate(top_models, 1):
        print(f"Rank {i}: Accuracy = {acc:.2f}%, Model Path = {path}")
    return top_models

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    test_accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")
    top_models = save_top_models(epoch, test_accuracy, model, top_models)

print("Finished Training")
