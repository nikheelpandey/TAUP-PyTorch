import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm 
import os
import time 

from models import *
from utils import *
from data import *
from loss import *


parser = argparse.ArgumentParser(description='SIMCLR')

uid = 'SimCLR'
dataset_name = 'CIFAR10C'
data = 'dataset'
features = 128
batch = 4
accumulation =4
epochs = 150
lr = 1e-3
use_cuda = True
device_id = 0
 

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Setup tensorboard
log_dir = "./tb" 

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')


logger = SummaryWriter(comment='_' +  uid + '_' + dataset_name)



