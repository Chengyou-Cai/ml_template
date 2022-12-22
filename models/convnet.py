import torch.nn as nn
import torchvision.models.resnet as resnet

class ConvNet(nn.Module):

    def __init__(self,num_classes=10) -> None:
        super(ConvNet,self).__init__()
        self.start_conv = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(1,1))
        self.backbone = resnet.resnet18(num_classes=num_classes)

    def forward(self,x):
        out = self.start_conv(x)
        out = self.backbone(out)
        return out