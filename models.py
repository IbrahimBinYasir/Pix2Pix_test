import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from utils_cv import my_data_set,showTensor
from clean_data import T_Data

d = T_Data()
train_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.5, 0.5, 0.5],
       std=[0.5, 0.5, 0.5]
    )
])
#d = train_preprocess(d)

train_dataloader = DataLoader(d, batch_size=5, shuffle=True)
train_image = next(iter(train_dataloader))
print(train_image.shape)
