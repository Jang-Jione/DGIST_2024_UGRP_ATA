from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from PIL import Image
import tqdm
import os
from peft import LoraConfig, get_peft_model


# Create directory for saving models
os.makedirs("top_models", exist_ok=True)

# 1. Model and processor loading
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model_weights_path = hf_hub_download(
#     repo_id="JANGJIWON/EmoSet118K_MonetStyle_CLIP_student",
#     filename="CLIPEmoset118k_mone.pth"
# )
# model.load_state_dict(torch.load(model_weights_path, map_location='cpu', weights_only=True), strict=False)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank of low-rank matrices
    lora_alpha=16,  # scaling factor for LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # LoRA will be applied to attention projections
    lora_dropout=0.1,  # dropout rate for LoRA layers
    bias="none",  # no bias terms in LoRA layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# 2. Dataset loading
train_dataset = load_dataset("xodhks/EmoSet118K", split="train")
test_dataset = load_dataset("xodhks/ugrp-survey-test")

# 3. Label mapping
possible_labels = ["Happiness", "Anger", "Surprise", "Disgust", "Fear", "Sadness"]

# 4. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Modified collate function
def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]

    # Convert labels to text
    label_texts = [possible_labels[label] if isinstance(label, int) else label for label in labels]

    # Process inputs
    inputs = processor(
        images=images,
        text=label_texts,
        return_tensors="pt",
        padding=True
    )

    # Store original labels for loss calculation
    inputs['original_labels'] = torch.tensor([possible_labels.index(text) for text in label_texts])

    return inputs

# 6. DataLoader setup
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True
)

# 7. Optimizer setup
optimizer = AdamW(model.parameters(), lr=1e-5)

# 8. Loss function
def compute_loss(logits_per_image, labels):
    # 대칭적 교차 엔트로피 손실
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size).to(logits_per_image.device)

    # 이미지-텍스트 간 대칭 손실
    loss_i = torch.nn.functional.cross_entropy(logits_per_image, targets)
    loss_t = torch.nn.functional.cross_entropy(logits_per_image.t(), targets)

    return (loss_i + loss_t) / 2

# 9. Model saving function
def save_top_models(epoch, accuracy, model, top_models):
    model_filename = f"model_epoch_{epoch + 1}_accuracy_{accuracy:.2f}.pth"
    model_path = os.path.join("top_models", model_filename)

    # Add new model to top_models list
    top_models.append((accuracy, model_path))
    top_models = sorted(top_models, key=lambda x: x[0], reverse=True)[:10]

    # Only save if model is in top 10
    if (accuracy, model_path) in top_models:
        torch.save(model.state_dict(), model_path)
        print("\nTop 10 Models (by accuracy):")
        for i, (acc, path) in enumerate(top_models, 1):
            print(f"Rank {i}: Accuracy = {acc:.2f}%, Model Path = {path}")

    return top_models

# 10. Training function (refactored)
# 10. Training function (refactored with incorrect examples logging)
def train(model, dataloader, optimizer, epochs=100):
    model.train()
    top_models = []
    best_accuracy = 0

    # Process the training loop
    for epoch in range(epochs):
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        incorrect_examples = []  # 잘못 분류된 예시를 저장할 리스트
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'original_labels'}
            labels = batch['original_labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            # Calculate probabilities and predictions
            probs = logits_per_image.softmax(dim=1)
            predictions = probs.argmax(dim=1)

            # Update accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Calculate loss
            loss = compute_loss(logits_per_image, labels)
            total_loss += loss.item()

            # Track incorrect examples
            for i in range(len(predictions)):
                if predictions[i] != labels[i]:  # 잘못 분류된 경우
                    incorrect_examples.append((inputs['pixel_values'][i], predictions[i].item(), labels[i].item()))

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update progress bar with loss and accuracy
            current_accuracy = (correct_predictions / total_predictions) * 100
            progress_bar.set_postfix({
                'loss': f'{total_loss/(progress_bar.n+1):.4f}',
                'accuracy': f'{current_accuracy:.2f}%'
            })

        # Epoch summary
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {total_loss/len(dataloader):.4f}")
        print(f"Train Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate the model on test dataset after each epoch
        test_accuracy = evaluate(model, test_dataset, processor, possible_labels, device)
        print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")

        # Save top models
        top_models = save_top_models(epoch, test_accuracy, model, top_models)
        print("-" * 50)

    return top_models

# 11. Evaluation function
def evaluate(model, dataset, processor, possible_labels, device):
    model.eval()
    correct_predictions = 0
    total = 0

    # 텍스트 프롬프트 생성
    text_inputs = [f"This image represents {l} emotion." for l in possible_labels]

    test_dataset = dataset['train']
    progress_bar = tqdm.tqdm(test_dataset, desc="Evaluating")

    for item in progress_bar:
        image = item['image']
        label = item['label']

        # Convert label to text if needed
        true_label = possible_labels[label] if isinstance(label, int) else label

        # Prepare inputs
        inputs = processor(
            images=image,
            text=text_inputs,  # 추가된 부분: 텍스트 프롬프트 사용
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            predicted_label = possible_labels[probs.argmax().item()]

        if predicted_label == true_label:
            correct_predictions += 1
        total += 1

        # Update progress bar
        progress_bar.set_postfix({'accuracy': f'{(correct_predictions/total)*100:.2f}%'})

    final_accuracy = (correct_predictions / total) * 100
    return final_accuracy

# 12. Main execution
print("Starting training...")
top_models = train(model, train_dataloader, optimizer, epochs=100)

print("\nEvaluating final model...")
test_accuracy = evaluate(model, test_dataset, processor, possible_labels, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Print final top 10 models
print("\nFinal Top 10 Models:")
for i, (acc, path) in enumerate(top_models, 1):
    print(f"Rank {i}: Accuracy = {acc:.2f}%, Model Path = {path}")
