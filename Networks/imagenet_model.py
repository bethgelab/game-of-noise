import torch.nn as nn
import torch
import torchvision.models as models


class ZeroOneResNet18(nn.Module):
    def __init__(self, device='cuda', pretrained=False):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)

    def forward(self, input):
        input = (input - self.mean) / self.std
        return self.resnet(input)
    
    
class ZeroOneInceptionV3(nn.Module):
    def __init__(self, device='cuda', pretrained=True):
        super().__init__()
        self.inception = models.inception_v3(pretrained=pretrained, transform_input=True)
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)

    def forward(self, input):
        input = (input - self.mean) / self.std
        return self.inception(input)   
    
    
class ZeroOneResNet50(nn.Module):
    def __init__(self, device='cuda', pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)

    def forward(self, input):
        input = (input - self.mean) / self.std
        return self.resnet(input)
    

class ZeroOneResNet152(nn.Module):
    def __init__(self, device='cuda', pretrained=True):
        super().__init__()
        self.resnet = models.resnet152(pretrained=pretrained)
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)

    def forward(self, input):
        input = (input - self.mean) / self.std
        return self.resnet(input)

    
class ZeroOneResNet50_robust(nn.Module):
    def __init__(self, robust_model, device='cuda'):
        super().__init__()
        self.resnet = robust_model
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
        
    def forward(self, input):
        input = (input - self.mean) / self.std
        return self.resnet(input)
