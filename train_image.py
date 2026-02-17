import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm

from models.image_model import DeepShieldImageModel
from utils.preprocess import DeepfakeImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DeepfakeImageDataset("datasets/images")
print("📂 Dataset Statistics")
print("Total samples:", len(dataset))
print("Class to Index Mapping:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print("Training samples   :", len(train_ds))
print("Validation samples :", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = DeepShieldImageModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

# 📊 Evaluation
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n📊 Evaluation Metrics")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1:", f1_score(all_labels, all_preds))
print("AUC:", roc_auc_score(all_labels, all_probs))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

torch.save(model.state_dict(), "models/image_model.pth")
print("Model saved.")
