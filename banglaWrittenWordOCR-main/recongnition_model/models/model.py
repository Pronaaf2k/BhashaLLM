import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class resnet34(nn.Module):
    def __init__(self):
        super(resnet34, self).__init__()

        self.model = pretrainedmodels.__dict__["resnet34"](pretrained = None)

        # Modify the first layer to accept 1 channel instead of 3
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.l0 = nn.Linear(512, 178)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 8)

    def forward(self, x):
        bs,_,_,_ = x.shape
        x = self.model.features(x)
        x =  F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)

        return l0, l1, l2