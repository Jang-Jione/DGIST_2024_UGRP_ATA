import torch
import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# 1. 모델 및 프로세서 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=6,  # 레이블 개수
    ignore_mismatched_sizes=True,
).to(device)

# 2. 데이터 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(image)

# 3. Custom Dataset 정의
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.label_encoder = LabelEncoder()
        labels = [item['label'] for item in dataset]
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        label = item['label']
        if self.transform:
            img = self.transform(img)
        label = self.label_encoder.transform([label])[0]
        return {"pixel_values": img, "labels": torch.tensor(label, dtype=torch.long)}

# 4. 데이터셋 로드 및 분리
dataset = load_dataset("JANGJIWON/UGRP_sketchset_textbook")
split_dataset = dataset["train"].train_test_split(test_size=0.8, seed=42)  # 80% train, 20% test
train_data = split_dataset["train"]
test_data = split_dataset["test"]

train_dataset = CustomDataset(train_data, transform=preprocess_image)
test_dataset = CustomDataset(test_data, transform=preprocess_image)

from torch.utils.data import DataLoader, RandomSampler

train_loader = DataLoader(train_dataset, batch_size=16, sampler=RandomSampler(train_dataset, generator=torch.Generator().manual_seed(42)), num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# 5. 학습 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# 6. 평가 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# 7. 학습 루프
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

# 8. 임베딩 추출 함수
def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].cpu().numpy()
            outputs = model(pixel_values=pixel_values)
            embeddings.append(outputs.logits.cpu().numpy())
            labels.extend(batch_labels)
    embeddings = np.vstack(embeddings)
    return embeddings, np.array(labels)

# Silhouette Score 계산 함수
def calculate_silhouette_score(tsne_embeddings, labels, n_clusters):
    # KMeans를 사용해 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_embeddings)
    score = silhouette_score(tsne_embeddings, cluster_labels)
    return score

# t-SNE 시각화 함수
def visualize_tsne(embeddings, labels, title, n_labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, ticks=range(n_labels), label="Emotion Labels")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

    return tsne_result

# 9. Train 데이터 시각화 및 Silhouette Score 계산
train_embeds, train_labels = extract_embeddings(model, train_loader)
train_tsne = visualize_tsne(train_embeds, train_labels, "t-SNE Visualization of Train Embeddings", n_labels=5)
train_silhouette_score = calculate_silhouette_score(train_tsne, train_labels, n_clusters=6)
print(f"Train Silhouette Score: {train_silhouette_score:.4f}")

# 10. Test 데이터 시각화 및 Silhouette Score 계산
test_embeds, test_labels = extract_embeddings(model, test_loader)
test_tsne = visualize_tsne(test_embeds, test_labels, "t-SNE Visualization of Test Embeddings", n_labels=5)
test_silhouette_score = calculate_silhouette_score(test_tsne, test_labels, n_clusters=6)
print(f"Test Silhouette Score: {test_silhouette_score:.4f}")
