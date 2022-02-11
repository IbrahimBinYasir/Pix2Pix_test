import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from utils_cv import my_data_set, showTensor, T_Normalize


def normal(img):
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
    img = trans(img)
    #img = np.asarray(img)
    return img

class T_Data(Dataset):

    def __init__(self, start=1, end=378, dir='./data/base', transform=None):
        self.all_data = my_data_set(start, end, dir)
        self.x_data = self.all_data[:, [0], :, :, :]
        self.x_data = np.squeeze(self.x_data)
        self.target = self.all_data[:, [1], :, :, :]
        self.target = np.squeeze(self.target)
        self.transform = transform

    def __getitem__(self, index):
        # x = torch.tensor(self.x_data[index])
        # y = torch.tensor(self.target[index])
        if self.transform:
            x_data = self.transform(self.x_data[index])
            target = self.transform(self.target[index])

        return x_data, target

    def __len__(self):
        return len(self.x_data)
