from src.models import Pretrain_model
from src.datasets import MEG_image_Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import set_seed
from torchmetrics import Accuracy
import numpy as np

seed = 1234
set_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

loader_args = {"batch_size": 128, "num_workers": 4}

train_set = MEG_image_Dataset("train", "data","data/train_image_paths.txt","image/Images/")
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = MEG_image_Dataset("val", "data","data/val_image_paths.txt","image/Images/")
val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

model = Pretrain_model(train_set.num_channels).to(device)  # 例としてクラス数を10に設定
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

min_val_loss = 100000
epochs = 20
for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, val_loss = [], []

        model.train()
        for X, y, subject_idxs,image_x in tqdm(train_loader, desc="Train"):
            X, y,image_x = X.to(device), y.to(device),image_x.to(device)

            y_pred = model(X)

            loss = criterion(y_pred, image_x)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        for X, y, subject_idxs, image_x in tqdm(val_loader, desc="Validation"):
            X, y,image_x = X.to(device), y.to(device), image_x.to(device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(criterion(y_pred, image_x).item())

        print(f"Epoch {epoch+1}/{epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f} ")
        torch.save(model.state_dict(), "outputs/pretrain/pretrain2_last.pt")

        if np.mean(val_loss) < min_val_loss:
            torch.save(model.state_dict(), "outputs/pretrain/pretrain2.pt")
            min_val_loss = np.mean(val_loss)