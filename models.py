import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models.convnext import LayerNorm2d
import torch.nn.functional as F
import timm

class DualInputResNet18(nn.Module):
    def __init__(self, base_model, num_classes=8):
        super().__init__()
        self.base_model = base_model
        # Remove the original fully connected layer
        self.base_model.fc = nn.Identity()
        # Create a new classifier layer for dual input
        self.fc = nn.Linear(base_model.num_features * 2, num_classes)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Process both inputs in parallel
        x1, x2 = self.base_model.conv1(x1), self.base_model.conv1(x2)
        x1, x2 = self.base_model.bn1(x1), self.base_model.bn1(x2)
        x1, x2 = self.base_model.act1(x1), self.base_model.act1(x2)
        x1, x2 = self.base_model.maxpool(x1), self.base_model.maxpool(x2)
        
        x1, x2 = self.base_model.layer1(x1), self.base_model.layer1(x2)
        x1, x2 = self.base_model.layer2(x1), self.base_model.layer2(x2)
        x1, x2 = self.base_model.layer3(x1), self.base_model.layer3(x2)
        x1, x2 = self.base_model.layer4(x1), self.base_model.layer4(x2)
        
        x1, x2 = self.base_model.global_pool(x1), self.base_model.global_pool(x2)
        
        if self.base_model.drop_rate:
            x1 = F.dropout(x1, p=float(self.base_model.drop_rate), training=self.training)
            x2 = F.dropout(x2, p=float(self.base_model.drop_rate), training=self.training)
        
        # Concatenate and classify
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

def create_dual_input_retinamnist_resnet18(num_classes=8, weightsFile=None):
    # Create base model
    model = timm.create_model('resnet18', pretrained=False, num_classes=5)
    
    # Load weights if provided
    if weightsFile is not None:
        state = torch.load(weightsFile, map_location='cpu')
        if 'model' in state:
            model.load_state_dict(state['model'], strict=False)
        elif 'net' in state:
            model.load_state_dict(state['net'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    
    # Wrap in dual input model
    dual_input_model = DualInputResNet18(model, num_classes=num_classes)
    return dual_input_model



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

class DualInputEfficientNetB4(nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(DualInputEfficientNetB4, self).__init__()
        self.efficientnet = models.efficientnet_b4(weights=weights)
        
        self.efficientnet.classifier[1] = nn.Linear(1792*2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.efficientnet.features(x1), self.efficientnet.features(x2)
        x1, x2 = self.efficientnet.avgpool(x1), self.efficientnet.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.efficientnet.classifier(x)
        return x
    
class DualInputEfficientNetB6(nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(DualInputEfficientNetB6, self).__init__()
        self.efficientnet = models.efficientnet_b6(weights=weights)
        
        self.efficientnet.classifier[1] = nn.Linear(2304*2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.efficientnet.features(x1), self.efficientnet.features(x2)
        x1, x2 = self.efficientnet.avgpool(x1), self.efficientnet.avgpool(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.efficientnet.classifier(x)
        return x
    
class DualInputEfficientNetB7(nn.Module):
    def __init__(self, num_classes=1000, weights=None):
        super(DualInputEfficientNetB7, self).__init__()
        self.efficientnet = models.efficientnet_b7(weights=weights)
        
        self.efficientnet.classifier[1] = nn.Linear(2560*2, num_classes)

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