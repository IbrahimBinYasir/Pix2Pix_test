import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch



def rand_crop(img, size):
    x = random.randint(0, img.shape[1] - size)
    y = random.randint(0, img.shape[0] - size)
    img = img[y:y + size, x:x + size]
    return img

def mirroring(img):
    if random.uniform(0, 1) > 0.5:
        img = cv2.flip(img, 1)
    return img

def T_Normalize(img):
    trans = transforms.Compose([transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
                                ])

    #trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img)
    #img = np.asarray(img)

    return img

def preprocess(image, jit, size, inter):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), inter)  # resizing the images
    image = cv2.resize(image, (jit, jit))  # resizing for jittering
    image = rand_crop(image, 256) # random_crop
    image = mirroring(image) #mirroring the image
    return image


def my_data_set(start, end, filepath):
    final = []
    for i in range (start,end+1):
        real = cv2.imread(filepath+"/cmp_b%04d.jpg"%i)
        real = preprocess(real, jit=286, size=256, inter=cv2.INTER_CUBIC)
        #real = Normalize(real)
        x_img = cv2.imread(filepath+"/cmp_b%04d.png"%i)
        x_img = preprocess(x_img, jit=286, size=256, inter=cv2.INTER_CUBIC)
        final.append([x_img, real])

    result = np.asarray(final)
    #result = final
    return result


def showTensor(aTensor):
    plt.figure()
    arr = aTensor.permute(2,1,0)
    arr = aTensor.numpy()
    plt.imshow(arr)
    plt.colorbar()
    plt.show()

