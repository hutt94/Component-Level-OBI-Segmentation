# -*- codeing = utf-8 -*-
# @Time : 2024/5/7 13:56
# @Author : 李昌杏
# @File : model.py
# @Software : PyCharm
import torch
import torch.nn as nn
from network.pvt import pvt_medium,PyramidVisionTransformer
from mmseg.models import build_head
import torchvision.models as models

# pvt_medium.pth can get from the PVT (PyramidVisionTransformer)
# the project home page is https://github.com/whai362/PVT

class Character_encoder(nn.Module):
    def __init__(self,num_classes,pvt_weigth='../pvt_medium.pth'):
        super(Character_encoder, self).__init__()
        self.backbone:PyramidVisionTransformer=pvt_medium(True,pvt_weigth)
        self.backbone.reset_classifier(num_classes)

        uper_block = dict(
            type='UPerHead',
            in_channels=[64,128,320,512],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

        self.upsample = build_head(uper_block)
        self.seg_head=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.classifier=models.resnet50()
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_features,num_classes)

    def forward(self,x,segmentation=True):
        t,scale=self.backbone.embedding(x) #class token, visual token ; 1x512, h x w x 512
        if segmentation:
            seg_logits = self.seg_head(self.upsample._forward_feature(scale))
            seg=torch.argmax(seg_logits,dim=1).unsqueeze(1).repeat(1,3,1,1).type(torch.float32)
            classify=self.classifier(seg)
            return seg_logits,classify
        return self.backbone.head(t),t #class logits, class_token

class Component_encoder(nn.Module):
    def __init__(self, num_classes, pvt_weigth='../pvt_medium.pth'):
        super(Component_encoder, self).__init__()
        self.backbone: PyramidVisionTransformer = pvt_medium(False, pvt_weigth)
        self.backbone.reset_classifier(num_classes)
        self.fc_IC=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,5))
        self.fc_CDC=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,3))
        self.fc_CC=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,3))

    def forward(self, x,stroke=True):
        cls, scale = self.backbone.embedding(x)

        if stroke==True:
            IC=self.fc_IC(cls)
            CDC=self.fc_CDC(cls)
            CC=self.fc_CC(cls)
            return IC,CDC,CC

        return self.backbone.head(cls),cls

class Baseline_encoder(nn.Module):
    def __init__(self, num_classes,):
        super(Baseline_encoder, self).__init__()
        self.classifier = models.resnet50()
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_features, num_classes)
        self.fc_IC=nn.Sequential(nn.LayerNorm(num_features),nn.Linear(num_features,5))
        self.fc_CDC=nn.Sequential(nn.LayerNorm(num_features),nn.Linear(num_features,3))
        self.fc_CC=nn.Sequential(nn.LayerNorm(num_features),nn.Linear(num_features,3))

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.classifier.conv1(x)
        x = self.classifier.bn1(x)
        x = self.classifier.relu(x)
        x = self.classifier.maxpool(x)

        x = self.classifier.layer1(x)
        x = self.classifier.layer2(x)
        x = self.classifier.layer3(x)
        x = self.classifier.layer4(x)

        x = self.classifier.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):

        fea = self._forward_impl(x)

        IC=self.fc_IC(fea)
        CDC=self.fc_CDC(fea)
        CC=self.fc_CC(fea)
        cls = self.classifier.fc(fea)
        return cls,IC,CDC,CC


if __name__ == '__main__':
    data=torch.zeros((4,3,224,224))
    net=Component_encoder(100)
    out=net(data)
    for _ in out:print(_.shape)
