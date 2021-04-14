import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm 
import os
import time
from datetime import datetime 
from knn_monitor import knn_monitor
from model import ContrastiveLearner
from dataset_loader import get_train_mem_test_dataloaders, gpu_train_transformer,gpu_test_transformer
from logger import Logger




uid = 'SimCLR'
dataset_name = 'cifar10'
data_dir = 'dataset'
ckpt_dir = "./ckpt"
features = 128
batch = 512
accumulation = 4
epochs = 3
lr = 1e-3
use_cuda = True
device_id = 0
image_size = (32,32)
wt_decay  = 0.99
 

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Setup tensorboard
log_dir = "./runs" 

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')
    
if not os.path.exists('ckpt'):
    os.makedirs('ckpt')


# logger = SummaryWriter(comment='_' +  uid + '_' + dataset_name)

logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)

train_loader, memory_loader, test_loader =  get_train_mem_test_dataloaders(dataset = dataset_name,
                                                        data_dir=data_dir, 
                                                        batch_size = batch, 
                                                        num_workers = 4,
                                                        download=True)
train_transform = gpu_train_transformer(image_size)
test_transform = gpu_test_transformer(image_size)



model = ContrastiveLearner().to(device)
optimizer = optim.Adam(model.parameters(), 
            lr=lr,
            weight_decay=False) 
scheduler = ExponentialLR(optimizer, gamma= 1)


min_loss = np.inf #ironic
accuracy = 0

# start training 
global_progress = tqdm(range(0, epochs), desc=f'Training')
data_dict = {"loss": 100}
for epoch in global_progress:
    model.train()   

    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
    for idx, (image, label) in enumerate(local_progress):
        
        image = image.to(device)
        aug_image = train_transform(image)

        model.zero_grad()
        loss = model.forward(image.to(device, non_blocking=True),aug_image.to(device, non_blocking=True))

        data_dict['loss'] = loss.item() 
        loss.backward()
        optimizer.step()
        scheduler.step()
        data_dict.update({'lr': scheduler.get_lr()[0]})
        local_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)
        
    accuracy = knn_monitor(model.backbone, memory_loader, test_loader, device, hide_progress=True) 
    epoch_dict = {'epoch':epoch, 'accuracy':accuracy}
    global_progress.set_postfix(epoch_dict)
    logger.update_scalers(epoch_dict)

model_path = os.path.join(ckpt_dir, f"{uid}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
torch.save({
    'epoch':epoch+1,
    'state_dict': model.module.state_dict()
        }, model_path)
print(f'Model saved at: {model_path}')