import torch
import torchvision
import torch.nn as nn
from torch import allclose
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.testing import assert_allclose
import kornia
from kornia import augmentation as K
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF
import sys
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt


CIFAR_MEAN =  [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  =  [0.24703233, 0.24348505, 0.26158768]


CIFAR_MEAN_ =  torch.FloatTensor([CIFAR_MEAN, CIFAR_STD])
CIFAR_STD_  =  torch.FloatTensor([CIFAR_MEAN, CIFAR_STD])


# GPU based augmentation: https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#scrollTo=-AnIAZjeIP35

class InitalTransformation():
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            # transforms.Normalize(CIFAR_MEAN_,CIFAR_STD_),
        ])

    def __call__(self, x):
        x = self.transform(x)
        return  x


def get_clf_train_test_dataloaders(dataset = "cifar10", percent_train_sample = 20,
                                  data_dir="./dataset", batch_size = 16,
                                    num_workers = 4, download=True):

    tr = torchvision.datasets.CIFAR10(data_dir, train=True, transform=InitalTransformation(), download=True)
    samples = list(range(0, int(len(tr)*percent_train_sample/100)))
    tr_subset =  torch.utils.data.Subset(tr, samples)

    train_loader = torch.utils.data.DataLoader(
            dataset = tr_subset,
            shuffle=True,
            batch_size= batch_size,
            num_workers = 4 )

    test_loader = torch.utils.data.DataLoader(
            dataset = torchvision.datasets.CIFAR10(data_dir, train=True,
                                                transform=InitalTransformation(), download=True),
            shuffle=True,
            batch_size= batch_size,
            num_workers = 4
        )

    return train_loader, test_loader


def get_train_mem_test_dataloaders(dataset = "cifar10", data_dir="./dataset", batch_size = 16,num_workers = 4, download=True): 
    
    train_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(data_dir, train=True, transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
    )
    
    memory_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(data_dir, train=False, transform=InitalTransformation(), download=download),
        shuffle=False,
        batch_size= batch_size,
        num_workers = num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(data_dir, train=False, transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
        )
    return train_loader, memory_loader, test_loader


def gpu_transformer(image_size,s=.2):
        
    train_transform = nn.Sequential(
                
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                # kornia.augmentation.RandomGrayscale(p=0.05),
            )

    test_transform = nn.Sequential(  
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                # kornia.augmentation.RandomGrayscale(p=0.05),
        )

    return train_transform , test_transform
                
def get_clf_train_test_transform(image_size,s=.2):
        
    train_transform = nn.Sequential(
                
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                # kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_),
            )

    test_transform = nn.Sequential(  
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                # kornia.augmentation.RandomGrayscale(p=0.05),
                # kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_)
        )

    return train_transform , test_transform




if __name__=="__main__":
    device = torch.device('cuda')
    print(f"Running with device: {device}")

    data_dir="./dataset" 
    batch_size = 16
    num_workers = 8
    download=True
    train = True
    image_size = (32,32)
    s = 1.0

    

    train_transform = nn.Sequential(
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.2,1.0)),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.5),
                kornia.augmentation.RandomGrayscale(p=0.2),
            )


    train_loader,_,_ = get_train_mem_test_dataloaders()
    
    from tqdm import tqdm
    local_progress = tqdm(train_loader, desc=f'Epoch {1}/{1}')

    for i , sample in enumerate(local_progress):
        images = sample[0].to(device)
        transformed_images = train_transform(images)         


