from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 1. 모델 및 프로세서 로드
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. 데이터셋 로드 및 분리
dataset = load_dataset("JANGJIWON/UGRP_sketchset_textbook")
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# 3. 레이블을 감정 텍스트로 매핑
possible_labels = ["Happiness", "Sadness", "Disgust", "Fear", "Anger", "Surprise"]

# 4. 데이터셋 처리 함수 정의
def collate_fn(samples):
    images = [s['image'] for s in samples]
    labels = [s['label'] for s in samples]
    inputs = processor(images=images, text=possible_labels, return_tensors="pt", padding=True)
    inputs['labels'] = torch.tensor(labels)
    return inputs

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 5. 학습 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5

# 6. 학습 및 평가 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # 이미지-텍스트 유사도
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# 7. 학습 루프
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits_per_image, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
    
    # Train 정확도
    train_acc = evaluate(model, train_loader)
    print(f"Train Accuracy: {train_acc:.2f}%")
    
    # Test 정확도
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

# 8. t-SNE 시각화를 위한 임베딩 추출
def extract_embeddings(model, loader):
    model.eval()
    image_embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            batch_labels = batch['labels'].numpy()
            outputs = model(**inputs)
            
            image_embeds = outputs.image_embeds.cpu().numpy()
            
            image_embeddings.append(image_embeds)
            labels.extend(batch_labels)
    
    image_embeddings = np.vstack(image_embeddings)
    return image_embeddings, np.array(labels)

# Silhouette Score 계산 함수
def calculate_silhouette_score(tsne_embeddings, labels, n_clusters):
    # KMeans를 사용해 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_embeddings)
    
    # Silhouette Score 계산
    score = silhouette_score(tsne_embeddings, cluster_labels)
    return score

# Train 데이터에서 임베딩 추출
image_embeds, embed_labels = extract_embeddings(model, train_loader)

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=42)
image_tsne = tsne.fit_transform(image_embeds)

# Train 데이터 Silhouette Score 계산
train_silhouette_score = calculate_silhouette_score(image_tsne, embed_labels, n_clusters=len(possible_labels))
print(f"Train Data Silhouette Score: {train_silhouette_score:.4f}")

# 시각화
plt.figure(figsize=(10, 7))
scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=embed_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=range(len(possible_labels)), label='Emotion Labels')
plt.title('t-SNE Visualization of Train Image Embeddings')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Test 데이터에서 임베딩 추출
test_image_embeds, test_embed_labels = extract_embeddings(model, test_loader)

# t-SNE 적용
perplexity = min(30, len(test_image_embeds) - 1)  # perplexity는 샘플 수보다 작아야 함
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

# Test 데이터에 t-SNE 적용
test_image_tsne = tsne.fit_transform(test_image_embeds)

# Test 데이터 Silhouette Score 계산
test_silhouette_score = calculate_silhouette_score(test_image_tsne, test_embed_labels, n_clusters=len(possible_labels))
print(f"Test Data Silhouette Score: {test_silhouette_score:.4f}")

# Test 데이터 시각화
plt.figure(figsize=(10, 7))
scatter = plt.scatter(test_image_tsne[:, 0], test_image_tsne[:, 1], c=test_embed_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=range(len(possible_labels)), label='Emotion Labels')
plt.title('t-SNE Visualization of Test Image Embeddings')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
