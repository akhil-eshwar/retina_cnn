import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight


IMG_SIZE = 224


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])


val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


class RetinaDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        img_id = self.df.loc[idx, "id_code"]
        label = self.df.loc[idx, "diagnosis"]

        img_path = os.path.join(self.img_dir, img_id + ".png")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


def get_sampler(train_df):

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_df["diagnosis"]),
        y=train_df["diagnosis"]
    )

    weights = train_df["diagnosis"].map(
        {i: w for i, w in enumerate(class_weights)}
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler