import os
import torch 
from tqdm import tqdm
import torch.nn as nn
from logger import Logger
import torch.optim as optim
from datetime import datetime
from model import FineTunedModel
from torchvision.models import resnet18
from model import ContrastiveModel, get_backbone
from dataset_loader import get_clf_train_test_transform
from dataset_loader import get_clf_train_test_dataloaders


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

backbone= get_backbone(resnet18(pretrained=False))
model = ContrastiveModel(backbone).to(device)
obj = torch.load("/home/octo/Desktop/clr/ckpt/0417151425/SimCLR_0417175433.pth")

model.load_state_dict(obj['state_dict'])


encoder  = model.backbone
last_layers = torch.nn.Sequential(*(list(model.projectionhead.children())[0:2]))

encoder = nn.Sequential(
            encoder,
            last_layers)

new_model = FineTunedModel(encoder,model.output_dim).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.0001,
                      momentum=0.99, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0.0
batch = batch_size = 128
uid = "ssc"               #second_stage_classifier
percent_train_sample = 15 #percentage of labeled data used for supervised training
epochs = 50


train_loader, test_loader = get_clf_train_test_dataloaders(percent_train_sample=percent_train_sample)
train_transform, test_transform = get_clf_train_test_transform(image_size = (32,32))


def train_classifier(epoch, epochs):
    new_model.train()
    train_loss = 0
    correct = 0
    total = 0

    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
    for idx, (images, labels) in enumerate(local_progress):
        images, labels = images.to(device), labels.to(device)
        images = images.to(device)
        aug_image = train_transform(images)

        optimizer.zero_grad()
        outputs = new_model(aug_image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        data_dict = {"loss": train_loss, "accuracy":100.*correct/total}
        local_progress.set_postfix(data_dict)

    return data_dict

def test_classifier(epoch, epochs):

    global best_acc
    new_model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():    
        local_progress = tqdm(test_loader, desc=f'Epoch {epoch}/{epochs}')
        for idx, (images, label) in enumerate(local_progress):
            
            images, label = images.to(device), label.to(device)
            images = test_transform(images)
            outputs = new_model(images)
            loss = criterion(outputs, label)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            data_dict = {"test_loss": test_loss, "test_accuracy":100.*correct/total}
            local_progress.set_postfix(data_dict)
            
        # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': new_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        model_path = os.path.join(ckpt_dir, f"{uid}_{datetime.now().strftime('%m%d%H%M%S')}.pth")

        torch.save({
            'epoch':epoch+1,
            'state_dict': new_model.state_dict()
                }, model_path)
        print(f'Model saved at: {model_path}')
        best_acc = acc

    return data_dict

ckpt_dir = "./ckpt/clf_"+str(datetime.now().strftime('%m%d%H%M%S'))
log_dir = "runs/clf_"+str(datetime.now().strftime('%m%d%H%M%S'))
logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)




for epoch in range(0, epochs):
    data_dict = train_classifier(epoch,epochs)
    logger.update_scalers(data_dict)
    data_dict = test_classifier(epoch,epochs)
    logger.update_scalers(data_dict)
    scheduler.step()