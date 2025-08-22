import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn




# class TinyVGG(nn.Module):
    
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
#         super().__init__()
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape, 
#                       out_channels=hidden_units, 
#                       kernel_size=3, # how big is the square that's going over the image?
#                       stride=1, # default
#                       padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units, 
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2) # default stride value is same as kernel_size
#         )
#         self.conv_block_2 = nn.Sequential(
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             # Where did this in_features shape come from? 
#             # It's because each layer of our network compresses and changes the shape of our input data.
#             nn.Linear(in_features=hidden_units*16*16,
#                       out_features=output_shape)
#         )
    
#     def forward(self, x: torch.Tensor):
#         x = self.conv_block_1(x)
#         # print(x.shape)
#         x = self.conv_block_2(x)
#         # print(x.shape)
#         x = self.classifier(x)
#         # print(x.shape)
#         return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet34Custom(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_shape, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, output_shape)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x