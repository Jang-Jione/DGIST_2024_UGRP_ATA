import torch
import os
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm  # tqdm를 사용해 진행 상태를 시각화

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# 1. 모델 및 프로세서 로드
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. 데이터셋 로드 및 분리
train_dataset = load_dataset("xodhks/EmoSet118K_MonetStyle", split="train")
test_dataset = load_dataset("xodhks/Children_Sketch", split="train")

# 3. 레이블을 감정 텍스트로 매핑 (CoT 방식으로)
possible_labels = ["Happiness", "Sadness", "Disgust", "Fear", "Anger", "Surprise"]

# 4. 데이터셋 처리 함수 정의 (CoT 방식 프롬프트 추가)
def collate_fn(samples):
    images = [s['image'] for s in samples]
    labels = [s['label'] for s in samples]
    
    # CoT 방식 프롬프트 작성: 감정을 추론하는 과정
    prompts = [f"Given this image, what emotion is being expressed? 1. Consider the intent behind the image. 2. Carefully analyze the features of the image. 3. Based on steps 1 and 2, predict the emotion expressed in the image. Possible emotions: 'Happiness', 'Sadness', 'Disgust', 'Fear', 'Anger', 'Surprise'."
               for _ in samples]  # 이미지마다 추론 프롬프트
    
    # Processor에 프롬프트와 이미지를 함께 전달
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    inputs['labels'] = torch.tensor(labels)
    return inputs

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank of low-rank matrices
    lora_alpha=16,  # scaling factor for LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # LoRA will be applied to attention projections
    lora_dropout=0.1,  # dropout rate for LoRA layers
    bias="none",  # no bias terms in LoRA layers
)

# 6. LoRA 적용 모델 준비
model = get_peft_model(model, lora_config)

# 7. 학습 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

model.to(device)

num_epochs = 100

# 8. 평가 함수
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

# 9. 모델 저장 함수
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    test_accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")
    top_models = save_top_models(epoch, test_accuracy, model, top_models)

print("Finished Training")
