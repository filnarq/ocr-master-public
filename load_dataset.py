import numpy as np
import os

import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

def get_classes(dataset_dir):
    return sorted(os.listdir(dataset_dir))

def load_dataset_torch(dataset_dir, resize=16, batch_size=32, show=False):
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=128.0),
        # transforms.CenterCrop(9),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if show:
        classes = get_classes(dataset_dir)
        images, labels = next(iter(dataloader))
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        imshow(utils.make_grid(images))
    
    return dataloader

def imshow(img, label=None):
    if label:
        print(label)
    img = np.transpose(img.numpy(), (1,2,0))
    plt.imshow(img)
    plt.show()
