from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
import torchvision

import torch
from PIL import ImageFilter
from PIL import Image


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]



class TransformSingles():
    def __init__(self, image_size=None, train=False):
        image_size = 224 if image_size is None else image_size 

        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
            ])

    def __call__(self, x):
        return self.transform(x)



class SimCLRTransform():
    def __init__(self, image_size=None, s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN,CIFAR_STD)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 
    

def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):

    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset

def get_train_memory_test_loaders(dataset = "cifar10", data_dir="./dataset", batch_size = 16, download=True): 
    
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(

            dataset=dataset,
            data_dir = data_dir, 
            transform = SimCLRTransform() , 
            train=True, 
            download=False
            ),

        shuffle=True,
        batch_size= batch_size,
    )

    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(

            dataset=dataset,
            data_dir = data_dir, 
            transform = TransformSingles() , 
            train=True, 
            download=False
            ),

        shuffle=False,
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(

            dataset=dataset,
            data_dir = data_dir, 
            transform = TransformSingles() , 
            train=False, 
            download=False
            ),


        shuffle=False,
        batch_size=batch_size,
    )

    return train_loader, memory_loader, test_loader