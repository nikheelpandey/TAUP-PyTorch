import torch
import torch.nn as nn
from loss import ContrastiveLoss
from torchvision.models import resnet50


def get_backbone(backbone, castrate=True):
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    return backbone

class ProjectionHead(nn.Module):
    def __init__(self,in_shape,out_shape=256):
        super().__init__()
        hidden_shape = in_shape//2

        self.layer_1 = nn.Sequential(
            nn.Linear(in_shape,hidden_shape),
            nn.ReLU(inplace=True)
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(hidden_shape,hidden_shape),
            nn.ReLU(inplace=True)
        )

        self.layer_3 = nn.Linear(hidden_shape,out_shape)

    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        return x


class ContrastiveLearner(nn.Module):
    def __init__(self, backbone=resnet50(), projection_head=None):
        super().__init__()

        self.backbone = get_backbone(backbone)
        self.projection_head = ProjectionHead(backbone.output_dim)
        self.loss = ContrastiveLoss(temp=0.5, normalize= True)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projection_head
        )

    def forward(self,x,x_):
        
        z   = self.encoder(x)
        z_  = self.encoder(x_)
        loss= self.loss(z,z_)
        
        return loss



if __name__=="__main__":
    print(resnet50().fc.in_features)
    # eval(f'{resnet50()}')
    # backbone = (resnet50().eval)
    # print(type(backbone))
    # print(backbone.fc.in_features)

