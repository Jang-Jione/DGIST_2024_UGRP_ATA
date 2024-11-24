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
possible_labels = [
    "To identify the emotion in the image, I will start by analyzing the facial expressions and body language. If the image shows a smile, relaxed posture, and joyful atmosphere, it is likely representing Happiness.",
    "To identify the emotion in the image, I will focus on signs of anger such as furrowed brows, clenched fists, or an aggressive stance. If these signs are prominent, the emotion is most likely Anger.",
    "To identify the emotion in the image, I will look for signs of surprise, such as wide eyes and an open mouth. If these features are present, the emotion is likely Surprise.",
    "To identify the emotion in the image, I will observe facial expressions and body language. If the image shows a disgusted look, with scrunched facial features or a turned-away posture, the emotion is most likely Disgust.",
    "To identify the emotion in the image, I will consider signs of fear such as wide eyes, tense posture, and nervous behavior. If these traits are visible, the emotion is likely Fear.",
    "To identify the emotion in the image, I will check for signs of sadness, such as downturned eyes or a slumped posture. If these are apparent, the emotion being expressed is likely Sadness."
]

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

# 5. LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank of low-rank matrices
    lora_alpha=16,  # scaling factor for LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "visual_projection"],  # LoRA will be applied to attention projections
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

# 9. top loss 모델 저장 함수 정의
def save_top_loss_models(epoch, train_loss, model, top_model=None):
    if top_model is None or train_loss < top_model['loss']:
        model_save_path = f"best_loss_model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch} with train loss {train_loss:.4f}")
        return {"epoch": epoch, "loss": train_loss, "model_path": model_save_path}
    return top_model

# 10. 학습 루프
top_model = None  # 최상의 모델을 저장할 변수
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits_per_image, labels)
        loss.backward()
        optimizer.step(
        optimizer.zero_grad()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
    
    # Train 정확도
    train_acc = evaluate(model, train_loader)
    print(f"Train Accuracy: {train_acc:.2f}%")
    
    # Test 정확도
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # top loss 모델 저장
    top_model = save_top_loss_models(epoch, avg_train_loss, model, top_model)

print("Finished Training")
