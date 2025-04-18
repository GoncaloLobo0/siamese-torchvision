import torch
import torchvision.models as models
import torch.nn as nn

class SiameseVGG11(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG11, self).__init__()
        self.vgg = models.vgg11(weights=weights)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(25088*2, 25088),
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
    
class SiameseVGG13(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG13, self).__init__()
        self.vgg = models.vgg13(weights=weights)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(25088*2, 25088),
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
    
class SiameseVGG16(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG16, self).__init__()
        self.vgg = models.vgg16(weights=weights)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(25088*2, 25088),
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
    
class SiameseVGG19(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG19, self).__init__()
        self.vgg = models.vgg19(weights=weights)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(25088*2, 25088),
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
    
class SiameseConvnext_base(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseConvnext_base, self).__init__()
        self.convnext = models.convnext_base(weights=weights)
        self.convnext.classifier[2] = nn.Linear(1024, num_classes)
        
        self.merge = nn.Sequential(
            nn.Linear(1024*2, 1024),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.convnext.features(x1), self.convnext.features(x2)
        x1, x2 = self.convnext.avgpool(x1), self.convnext.avgpool(x2)
        
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.merge(x)
        
        x.unsqueeze(-1).unsqueeze(-1) # Fix the shape for classifier because it was already flattened for the merge layer
        
        x = self.convnext.classifier(x)
        return x
        