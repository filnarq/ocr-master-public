import numpy as np
import os

import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

def get_classes(dataset_dir):
    return sorted(os.listdir(dataset_dir))

def load_dataset_torch(dataset_dir, resize=24, crop=24, batch_size=32, show=False):
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.RandomCrop(crop+3, padding=3, padding_mode="reflect"),
        transforms.Resize((resize,resize)),
        transforms.RandomInvert(p=0.5),
        transforms.ColorJitter(hue=(-0.5,0.5), saturation=0.5),
        transforms.ToTensor()
    ])
    print(transform)
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

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
