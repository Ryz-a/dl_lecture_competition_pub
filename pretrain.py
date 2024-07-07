from src.models import Pretrain_Classifier
from src.datasets import MEG_image_Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import set_seed
from torchmetrics import Accuracy
import numpy as np
import wandb

seed = 1234
set_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

loader_args = {"batch_size": 128, "num_workers": 4}

train_set = MEG_image_Dataset("train", "data","data/train_image_paths.txt","image/images/")
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = MEG_image_Dataset("val", "data","data/val_image_paths.txt","image/images/")
val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

model = Pretrain_Classifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(device)  # 例としてクラス数を10に設定
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

max_val_acc = 0
accuracy = Accuracy(
    task="multiclass", num_classes=train_set.num_classes, top_k=10
).to(device)

for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs,image_x in tqdm(train_loader, desc="Train"):
            X, y,image_x = X.to(device), y.to(device),image_x.to(device)

            y_pred = model(X,image_x)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs, image_x in tqdm(val_loader, desc="Validation"):
            X, y,image_x = X.to(device), y.to(device), image_x.to(device)
            
            with torch.no_grad():
                y_pred = model(X,image_x)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), "drive/MyDrive/Colab Notebooks/DLBasics2024_colab/final/dl_lecture_competition_pub/outputs/pretrain/pretrain_last.pt")
        wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            torch.save(model.state_dict(), "drive/MyDrive/Colab Notebooks/DLBasics2024_colab/final/dl_lecture_competition_pub/outputs/pretrain/pretrain.pt")
            max_val_acc = np.mean(val_acc)