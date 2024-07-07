import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torchvision import transforms
from PIL import Image


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

def load_image_paths(txt_file,data_count):
    image_paths = []
    with open(txt_file, 'r') as file:
        for i in range(data_count):
            line = file.readline().strip()
            image_paths.append(line)
    return image_paths

class MEG_image_Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data",image_txt_file: str = "train_data.txt", image_root_dir: str = "data/images/", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.image_data_dir = image_root_dir
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        self.image_paths = load_image_paths(image_txt_file, self.X.shape[0])
        self.transform = transforms.ToTensor()#transform or transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        


        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(self.image_data_dir+image_path)
        image = image.resize((224,224))
        image = self.transform(image)
        #if self.transform:
        #    image = self.transform(image)

        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i],image
        else:
            return self.X[i], self.subject_idxs[i],image
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
