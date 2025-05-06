import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models.convnext import LayerNorm2d

class DualInputEfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(DualInputEfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=weights)
        
        self.efficientnet.classifier[1] = nn.Linear(1280*2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.efficientnet.features(x1), self.efficientnet.features(x2)
        x1, x2 = self.efficientnet.avgpool(x1), self.efficientnet.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.efficientnet.classifier(x)
        return x

class SiameseVGG11(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG11, self).__init__()
        self.vgg = models.vgg11(weights=weights)
        
        self.vgg.classifier[0] = nn.Linear(25088*2, 4096)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vgg.features(x1), self.vgg.features(x2)
        x1, x2 = self.vgg.avgpool(x1), self.vgg.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.vgg.classifier(x)
        return x
    
class SiameseVGG13(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG13, self).__init__()
        self.vgg = models.vgg13(weights=weights)
        
        self.vgg.classifier[0] = nn.Linear(25088*2, 4096)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vgg.features(x1), self.vgg.features(x2)
        x1, x2 = self.vgg.avgpool(x1), self.vgg.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.vgg.classifier(x)
        return x
    
class SiameseVGG16(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG16, self).__init__()
        self.vgg = models.vgg16(weights=weights)
        
        self.vgg.classifier[0] = nn.Linear(25088*2, 4096)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vgg.features(x1), self.vgg.features(x2)
        x1, x2 = self.vgg.avgpool(x1), self.vgg.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.vgg.classifier(x)
        return x
    
class SiameseVGG19(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVGG19, self).__init__()
        self.vgg = models.vgg19(weights=weights)
        
        self.vgg.classifier[0] = nn.Linear(25088*2, 4096)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vgg.features(x1), self.vgg.features(x2)
        x1, x2 = self.vgg.avgpool(x1), self.vgg.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.vgg.classifier(x)
        return x
    
class SiameseConvnext_base(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseConvnext_base, self).__init__()
        self.convnext = models.convnext_base(weights=weights)
        
        self.convnext.classifier[0] = LayerNorm2d(1024*2, eps=1e-06, elementwise_affine=True)
        self.convnext.classifier[2] = nn.Linear(1024*2, num_classes)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.convnext.features(x1), self.convnext.features(x2)
        x1, x2 = self.convnext.avgpool(x1), self.convnext.avgpool(x2)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.convnext.classifier(x)
        return x

class SiameseVit_b_32(torch.nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(SiameseVit_b_32, self).__init__()
        self.vit = models.vit_b_32(weights=weights)
        
        #self.vit.norm_layer = nn.LayerNorm(768*2, eps=1e-06, elementwise_affine=True)
        self.vit.heads.head = nn.Linear(768*2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.vit._process_input(x1), self.vit._process_input(x2)
        n = x1.shape[0]
        
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x1, x2 = torch.cat([batch_class_token, x1], dim=1), torch.cat([batch_class_token, x2], dim=1)
        x1, x2 = self.vit.encoder(x1), self.vit.encoder(x2)
        
        x1, x2 = x1[:, 0], x2[:, 0]
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.vit.heads(x)
        
        return x