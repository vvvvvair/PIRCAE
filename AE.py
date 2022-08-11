import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class R_AE1(nn.Module):
    def __init__(self,block):
        super(R_AE1, self).__init__()
        self.in_planes = 32 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2_3 = self._make_layer(block,32,1,stride=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.layer_transconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.layer_transconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer_transconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.Tanh()
        ) 
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))  
        self.in_planes = planes
        return nn.Sequential(*layers)

    def encoder(self,x):
        out = self.layer1(x)
        out = self.maxpool1(out)
        out = self.layer2_3(out)
        out = self.layer4(out)
        out = self.maxpool2(out)
        return out
    def decoder(self,x):
        out = self.layer_transconv1(x)
        out = self.layer_transconv2(out)
        out = self.layer_transconv3(out)
        return out
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
