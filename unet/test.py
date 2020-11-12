from unet import UNet
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from dataset import XPlaneDataset
from loss import dice_loss
from collections import defaultdict
import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


VAL_DATA_FOLDER = '/media/sourish/datadrive/datasets/flying_object_detection/xplane/city_all/washington_morning_broken_1_temp3'


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = np.clip(inp, 0, 1)
    # inp = (inp * 255).astype(np.uint8)
    
    return inp


if __name__ == "__main__":
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('unet.pth')
    net = net.to('cuda:0')
    net.eval()
    print('model: ', net)

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = XPlaneDataset(VAL_DATA_FOLDER, transform=tf)
    dataloader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)

    inputs, masks = next(iter(dataloader))
    print(inputs.shape, masks.shape)

    plt.figure()
    plt.imshow(reverse_transform(inputs[0]))

    inputs = inputs.to('cuda:0')
    masks = masks.to('cuda:0')
    pred = net(inputs)
    pred = pred.data.cpu().numpy()

    print(np.max(pred))

    plt.show()