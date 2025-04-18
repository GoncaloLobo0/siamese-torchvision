import torch
import torchvision.models as models
import torch.nn as nn

class SiameseVGG11(torch.nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(SiameseVGG11, self).__init__()
        self.vgg = models.vgg11(pretrained=pretrained)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(25088*2, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vgg.features(x1), self.vgg.features(x2)
        x1, x2 = self.vgg.avgpool(x1), self.vgg.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.merge(x)
        
        x = self.vgg.classifier(x)
        return x