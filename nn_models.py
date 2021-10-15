import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_resnet18_classifier(output_size, pretrained=True):

    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, output_size)
    return model_ft

def get_torchvision_model(output_size, pretrained=True):

    model_ft = models.mobilenet_v2(pretrained=True)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, output_size)
    return model_ft




class CNN_Net_32(nn.Module):
    def __init__(self, output_size, num_input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_Net(nn.Module):
    def __init__(self, output_size, num_input_channels=1):
        super().__init__()
        # Inp;ut 128x128
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
        # device=None, dtype=None)

        self.conv1 = nn.Conv2d(num_input_channels, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 5, 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(45, output_size)
        #self.pad = lambda x: F.pad(x, (1, 1, 1, 1))

    def forward(self, x):

        pad = lambda x: F.pad(x, (1, 1, 1, 1))
        f = F.tanh # or F.sigmoid

        x = self.pool(f(pad(self.conv1(x))))
        x = self.pool(f(pad(self.conv2(x))))
        x = self.pool(f(pad(self.conv3(x))))
        x = self.pool(f(pad(self.conv4(x))))
        x = pad(self.conv5(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("x.shape:", x.shape)
        x = self.fc(x)
        return x


#def get_small_cnn_classifier(output_size):
#    net = NN_Net(output_size)
#    return net
