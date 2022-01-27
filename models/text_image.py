import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TextImage(nn.Module):
    def __init__(self, freeze_feat_extractor=True, p=0.25, word_dim=300):
        super().__init__()
        feat_extractor = torchvision.models.resnet18(pretrained=True)
        feat_extractor.fc = nn.Identity()
        self.feat_extractor = feat_extractor
        if freeze_feat_extractor:
            for param in feat_extractor.parameters():
                param.requires_grad = False
        self.batchnorm = nn.BatchNorm1d(512 + word_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(512 + word_dim, 1)

    def forward(self, batch):
        img = batch['img']
        text = batch['text']
        text = text.squeeze(1) 

        img = self.feat_extractor(img)
        x = torch.cat((img, text), 1)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class TextImage2(nn.Module):
    def __init__(self, freeze_feat_extractor=True, p=0.25, word_dim=300):
        super().__init__()
        feat_extractor = torchvision.models.resnet50(pretrained=True)
        feat_extractor.fc = nn.Identity()
        self.feat_extractor = feat_extractor
        if freeze_feat_extractor:
            for param in feat_extractor.parameters():
                param.requires_grad = False
        self.batchnorm = nn.BatchNorm1d(2048 + word_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(2048 + word_dim, 1)

    def forward(self, batch):
        img = batch['img']
        text = batch['text']
        text = text.squeeze(1) 

        img = self.feat_extractor(img)
        x = torch.cat((img, text), 1)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class TextImage3(nn.Module):
    def __init__(self, freeze_feat_extractor=True, p=0.25, word_dim=300):
        super().__init__()
        feat_extractor = torchvision.models.resnet50(pretrained=True)
        feat_extractor.fc = nn.Identity()
        self.feat_extractor = feat_extractor
        if freeze_feat_extractor:
            for param in feat_extractor.parameters():
                param.requires_grad = False
        self.img_proj = nn.Linear(2048, word_dim)
        self.batchnorm = nn.BatchNorm1d(word_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(word_dim, 1)

    def forward(self, batch):
        img = batch['img']
        text = batch['text']
        text = text.squeeze(1) 

        img = self.feat_extractor(img)
        img = self.img_proj(img)
        x = img * text
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x