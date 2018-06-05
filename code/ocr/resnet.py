# Implementation of https://arxiv.org/pdf/1512.03385.pdf.
# See section 4.2 for model architecture on CIFAR-10.
# Some part of the code was referenced below.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2,3], num_classes=10, args=None):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0], 2)
        self.layer2 = self.make_layer(block, 64, layers[0], 2)
        self.layer3 = self.make_layer(block, 128, layers[0], 2)
        self.layer4 = self.make_layer(block, 128, layers[0], 2)
        self.layer5 = self.make_layer(block, 128, layers[0], 2)
        self.fc = nn.Linear(128, num_classes)

        # detect
        self.convt1 = nn.Sequential(
                nn.ConvTranspose2d(128,128,kernel_size=2, stride=2), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        self.convt2 = nn.Sequential(
                nn.ConvTranspose2d(128,128,kernel_size=2, stride=2), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        self.convt3 = nn.Sequential(
                nn.ConvTranspose2d(128,128,kernel_size=2, stride=2), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        self.convt4 = nn.Sequential(
                nn.ConvTranspose2d(128,128,kernel_size=2, stride=2), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        self.in_channels = 256
        self.dec1 = self.make_layer(block, 128, layers[0])
        self.in_channels = 256
        self.dec2 = self.make_layer(block, 128, layers[0])
        self.in_channels = 192
        self.dec3 = self.make_layer(block, 128, layers[0])
        self.in_channels = 160
        # self.dec4 = self.make_layer(block, 1, layers[0])
        self.dec4 = nn.Sequential(
                nn.Conv2d(160, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1, bias=True)
                )
        self.in_channels = 256
        # self.dec2 = self.make_layer(block, 256, layers[0])
        # self.output = conv3x3(256, 4 * len(args.anchors))
        self.bbox = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 4 * len(args.anchors), kernel_size=1, bias=True)
                )
        self.sigmoid = nn.Sigmoid()

        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, phase='train'):
        out = self.conv(x)
        # print out.size()
        out = self.bn(out)
        # print out.size()
        out = self.relu(out)
        # print out.size()
        out1 = self.layer1(out)     # 64
        # print out1.size()
        out2 = self.layer2(out1)    # 32
        # print out2.size()
        out3 = self.layer3(out2)    # 16
        # print out3.size()
        out4 = self.layer4(out3)    # 8
        # print out4.size()
        out5 = self.layer5(out4)    # 4
        # print out5.size()

        # out = F.adaptive_max_pool2d(out5, output_size=(1,1)).view(out.size(0), -1) # 128
        # out = out.view(out.size(0), -1)

        if phase == 'seg':
            out = F.adaptive_max_pool2d(out5, output_size=(1,1)).view(out.size(0), -1) # 128
            out = self.fc(out)
            out = out.view(out.size(0), -1)
        else:
            out = F.max_pool2d(out5, 2)
            out_size = out.size()
            # out = out.view(out_size[0],out_size[1],out_size[3]).transpose(1,2).contiguous().view(-1, out_size[1])
            out = out.view(out_size[0],out_size[1],out_size[2] * out_size[3]).transpose(1,2).contiguous().view(-1, out_size[1])
            out = self.fc(out)
            out = out.view(out_size[0], out_size[2] * out_size[3], -1).transpose(1,2).contiguous()
            out = F.adaptive_max_pool1d(out, output_size=(1)).view(out_size[0], -1)

        # print out.size()
        if phase not in ['seg', 'pretrain', 'pretrain2']:
            return out

        # detect
        cat1 = torch.cat([self.convt1(out5), out4], 1)
        # print cat1.size()
        dec1 = self.dec1(cat1)
        # print dec1.size()
        # print out3.size()
        cat2 = torch.cat([self.convt2(dec1), out3], 1) 
        # print cat2.size()
        dec2 = self.dec2(cat2)
        cat3 = torch.cat([self.convt3(dec2), out2], 1)
        dec3 = self.dec3(cat3)
        cat4 = torch.cat([self.convt4(dec3), out1], 1)
        seg = self.dec4(cat4)
        seg = seg.view((seg.size(0), seg.size(2), seg.size(3)))
        seg = self.sigmoid(seg)
        
        bbox = self.bbox(cat2)
        # dec2 = self.output(dec2)
        # print dec2.size()
        size = bbox.size()
        bbox = bbox.view((size[0], size[1], -1)).transpose(1,2).contiguous()
        bbox = bbox.view((size[0], size[2],size[3],-1, 4))

        return out, bbox, seg
    
# resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
