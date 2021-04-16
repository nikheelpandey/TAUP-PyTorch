import torch
import torch.nn as nn
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

# TAUP part
class ContrastiveModel(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.output_dim = 256
        
        self.backbone=backbone
        self.projectionhead= ProjectionHead(backbone.output_dim)
        
        self.encoder = nn.Sequential(
                        self.backbone,
                        self.projectionhead,
                        )
        
    def forward(self,x):
        z   = self.encoder(x)
        return z


# Fine Tunning
class FineTunedModel(nn.Module):

    def __init__(self, num_classes=10, encoder=None):
        super().__init__()
        self.num_classes = num_classes
        encoder = encoder
        fc = nn.Linear(encoder.output_dim, self.num_classes)
        self.model = nn.Sequential(
            encoder,
            fc
            )
        
  
    def forward(self, image):
        logits = self.model(image)
        return logits
        


if __name__=="__main__":
    print(resnet50().fc.in_features)
    eval(f'{resnet50()}')()
    backbone = (resnet50().eval)
    print(type(backbone))
    print(backbone.fc.in_features)

