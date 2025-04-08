import os
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn

# ✅ Paths
train_features_path = "/Users/abhigna/saved_features/train_features1.pkl"
val_features_path = "/Users/abhigna/saved_features/val_features1.pkl"
model_save_path = "/Users/abhigna/saved_features/graphsage_model.pth"

# ✅ Load Extracted Features
print("🔄 Loading extracted features...")
with open(train_features_path, "rb") as f:
    train_features, train_labels = pickle.load(f)

with open(val_features_path, "rb") as f:
    val_features, val_labels = pickle.load(f)

print(f"✅ Loaded Train Features: {train_features.shape}, Train Labels: {train_labels.shape}")
print(f"✅ Loaded Validation Features: {val_features.shape}, Validation Labels: {val_labels.shape}")

# ✅ Construct KNN Graph (k=20)
print("🔄 Constructing KNN Graphs...")
k = 20
adj_train_sparse = kneighbors_graph(train_features.numpy(), k, mode="connectivity", include_self=True)
adj_val_sparse = kneighbors_graph(val_features.numpy(), k, mode="connectivity", include_self=True)

def sparse_to_edge_index(sparse_mat):
    coo = coo_matrix(sparse_mat)
    return torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

train_edge_index = sparse_to_edge_index(adj_train_sparse)
val_edge_index = sparse_to_edge_index(adj_val_sparse)

# ✅ Create Graph Data Objects
train_data = Data(x=torch.tensor(train_features.numpy(), dtype=torch.float32), edge_index=train_edge_index, y=train_labels)
val_data = Data(x=torch.tensor(val_features.numpy(), dtype=torch.float32), edge_index=val_edge_index, y=val_labels)

print(f"✅ Created Train Graph Data: {train_data}")
print(f"✅ Created Validation Graph Data: {val_data}")

# ✅ Define GraphSAGE Model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):  # fixed constructor
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = pyg_nn.SAGEConv(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = pyg_nn.SAGEConv(256, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(F.relu(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(F.relu(x))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

# ✅ Model Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphSAGE(in_channels=train_features.shape[1], out_channels=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

train_data.to(device)
val_data.to(device)

# ✅ Training Loop
epochs = 100
best_val_acc = 0
train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []

print("🔄 Training GraphSAGE Model...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_out = model(train_data)
    train_loss = criterion(train_out, train_data.y)
    train_loss.backward()
    optimizer.step()

    _, train_pred = torch.max(train_out, dim=1)
    train_acc = accuracy_score(train_data.y.cpu(), train_pred.cpu())

    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_loss = criterion(val_out, val_data.y)
        _, val_pred = torch.max(val_out, dim=1)
        val_acc = accuracy_score(val_data.y.cpu(), val_pred.cpu())

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)

print(f"✅ Training Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")

# ✅ Load Best Model
model.load_state_dict(torch.load(model_save_path))
model.eval()

# ✅ Get Predictions
with torch.no_grad():
    val_out = model(val_data)
    _, val_pred = torch.max(val_out, dim=1)

# ✅ Compute Confusion Matrix
cm = confusion_matrix(val_data.y.cpu(), val_pred.cpu())

# ✅ Plot Confusion Matrix (Improved)
plt.figure(figsize=(10, 8))  # Bigger and clearer
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[f"Class {i}" for i in range(4)],
            yticklabels=[f"Class {i}" for i in range(4)],
            cbar_kws={"shrink": 0.8, "label": "Count"})

plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.title("Confusion Matrix", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# ✅ Print Classification Report
print("\nClassification Report:\n")
print(classification_report(val_data.y.cpu(), val_pred.cpu()))

# ✅ Plot Accuracy & Loss Graphs
plt.figure(figsize=(10, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy", marker="o")
plt.plot(val_accuracies, label="Validation Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Epoch vs Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss", marker="o", color="red")
plt.plot(val_losses, label="Validation Loss", marker="s", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Epoch vs Loss")

plt.tight_layout()
plt.show()
