import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # CIFAR-10 images are 3x32x32, flatten them to 3072-dimensional vectors
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10) # 10 classes in CIFAR-10
    
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp_net = MLPNet()

class CNNNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.2) 
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 3 * 3, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.gelu(self.conv1(x)))
        x = self.dropout(self.bn2(F.gelu(self.conv2(x))))
        x = self.pool(self.bn3(F.gelu(self.conv3(x))))
        x = self.dropout(self.bn4(F.gelu(self.conv4(x))))
        x = self.bn5(F.gelu(self.conv5(x)))
        x = self.dropout(self.pool(self.bn6(F.gelu(self.conv6(x)))))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

cnn_net2 = CNNNet2()