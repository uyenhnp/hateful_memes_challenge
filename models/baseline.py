import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5) # 124, 124, 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 62, 62, 32
        self.conv2 = nn.Conv2d(32, 64, 3) # 60, 60, 64 # after pooling: 30, 30, 64
        self.conv3 = nn.Conv2d(64, 128, 3) # 28, 28, 128
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) # 128,1
        self.fc = nn.Linear(128, 1)

    def forward(self, batch):
        img = batch['img']
        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc(x))
        return x

class Baseline2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(64, 128, 3) 
        self.conv3 = nn.Conv2d(128, 256, 3) 
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(256, 1)

    def forward(self, batch):
        img = batch['img']
        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc(x))
        return x


class Baseline3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1)

    def forward(self, batch):
        img = batch['img']
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x


class Baseline4(nn.Module):
    def __init__(self, freeze_feat_extractor=True, p=0.25):
        super().__init__()
        feat_extractor = torchvision.models.resnet18(pretrained=True)
        feat_extractor.fc = nn.Identity()
        self.feat_extractor = feat_extractor
        if freeze_feat_extractor:
            for param in feat_extractor.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(512, 1)

    def forward(self, batch):
        img = batch['img']
        x = self.feat_extractor(img)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class Baseline5(nn.Module):
    def __init__(self, freeze_feat_extractor=True, p=0.25):
        super().__init__()
        feat_extractor = torchvision.models.resnet50(pretrained=True)
        feat_extractor.fc = nn.Identity()
        self.feat_extractor = feat_extractor
        if freeze_feat_extractor:
            for param in feat_extractor.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(2048, 1)

    def forward(self, batch):
        img = batch['img']
        x = self.feat_extractor(img)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x