import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import timm
import torch.nn as nn
import numpy as np
import pickle  
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

# ✅ Paths
dataset_path = "./augmented_db"
save_dir = "./saved_features"
os.makedirs(save_dir, exist_ok=True)
train_features_path = os.path.join(save_dir, "train_features1.pkl")
val_features_path = os.path.join(save_dir, "val_features1.pkl")
model_save_path = os.path.join(save_dir, "best_graphsage_model.pth")

# ✅ Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Load Dataset
print("📂 Loading dataset...")
dataset = ImageFolder(root=dataset_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"✅ Dataset loaded. Train: {train_size} | Validation: {val_size}")

# ✅ Create DataLoaders (Fix multiprocessing issue)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers=0 to avoid errors
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ✅ Load Pretrained ViT Model
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)
vit_model.head = nn.Identity()
vit_model.eval()
print("✅ ViT Model Loaded.")

# ✅ Feature Extraction in Batches
def extract_features(dataloader, save_path):
    if os.path.exists(save_path):
        print(f"✅ Loading features from {save_path}")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    features, labels = [], []
    print(f"🔄 Extracting features and saving to {save_path}")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            output = vit_model(images)
            features.append(output.cpu())
            labels.append(targets.cpu())

            if batch_idx % 10 == 0:
                with open(save_path, "wb") as f:
                    pickle.dump((torch.cat(features), torch.cat(labels)), f)
                print(f"✅ Saved batch {batch_idx} progress")

    features, labels = torch.cat(features), torch.cat(labels)
    with open(save_path, "wb") as f:
        pickle.dump((features, labels), f)
    print(f"✅ Final features saved to {save_path}")
    return features, labels

# ✅ Extract Features
print("📥 Extracting training features...")
train_features, train_labels = extract_features(train_loader, train_features_path)
print("📥 Extracting validation features...")
val_features, val_labels = extract_features(val_loader, val_features_path)

# ✅ Reduce Feature Dimensionality using Incremental PCA
print("🔄 Applying Incremental PCA...")
pca = IncrementalPCA(n_components=256)
train_features_np = pca.fit_transform(train_features.numpy())
val_features_np = pca.transform(val_features.numpy())
print("✅ PCA transformation complete.")
