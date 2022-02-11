import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from utils_cv import my_data_set,showTensor

class TensorData(Dataset):

    def __init__(self, start = 1, end = 37, dir='./data/base'):

        self.x_data = torch.tensor(my_data_set(start,end,dir))
        transform_list = [
                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        x = torch.tensor(self.x_data[index])
        return x

    def __len__(self):
        return len(self.x_data)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.5, 0.5, 0.5],
       std=[0.5, 0.5, 0.5]
    )
])
dataset = my_data_set(1, 37, './data/base')
#t_dataset = torch.tensor(dataset)
print(dataset.shape)
# cv2.imshow('test', dataset[0])
# cv2.waitKey()
my = dataset[:,[0],:,:]
my = np.squeeze(my)
print(type(my),my.shape)
#img_test = t_dataset[15]
#okay = TensorData() #Using the defined dataset
first = dataset[1]
print(type(first),first.shape)
#first = first.numpy()
#a = torch.tensor(first, dtype=torch.float32)
a = first[0]
print(type(a))

plt.figure()
arr = a
#arr = arr.numpy()
print(np.min(arr))
print(type(arr), arr.shape)
plt.imshow(arr)  # normalization cuz of changing data types
plt.show()
#showTensor(a)

