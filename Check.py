import torch
from config import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from GEN import Generator
from DIS import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms
from clean_data import T_Data
from torchvision.utils import save_image

gen = Generator(in_channels=3, features=64).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))


trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
dataset = T_Data(transform=trans)
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), (len(dataset)-int(len(dataset)*0.8))])
train_loader = DataLoader(train_set,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)

val_dataset = val_set
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

x, y = next(iter(val_loader))
x, y = x.to(config.DEVICE), y.to(config.DEVICE)
model = load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
gen.eval()

with torch.no_grad():
    y_fake = gen(x)
    y_fake = y_fake * 0.5 + 0.5  # remove normalization#
    save_image(y_fake, 'my_eval' + f"/y_gen.png")
    save_image(x * 0.5 + 0.5, 'my_eval' + f"/input.png")