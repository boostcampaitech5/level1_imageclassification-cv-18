import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True).eval()
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        # for param, weight in self.resnet.named_parameters():
        #     if "fc" not in param:
        #         weight.requires_grad_(requires_grad=False)

    def forward(self, x):
        return self.resnet(x)


class MultiLabelResNet18(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelResNet18, self).__init__()
        num_classes = 8
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        # for param, weight in self.resnet.named_parameters():
        #     if "fc" in param:
        #         weight.requires_grad_(requires_grad=True)

    def forward(self, x):
        x = self.resnet(x)
        return torch.split(x, [3, 2, 3], dim=1)


class VitSmall224(nn.Module):
    def __init__(self, num_classes):
        super(VitSmall224, self).__init__()
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class MultiLabelCustomNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelCustomNet, self).__init__()
        self.resnet_mask = resnet18(pretrained=True)
        self.resnet_mask.fc = nn.Identity()
        self.resnet_gender = resnet18(pretrained=True)
        self.resnet_gender.fc = nn.Identity()
        self.resnet_age = resnet18(pretrained=True)
        self.resnet_age.fc = nn.Identity()
        self.fc_mask = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3),
        )
        self.fc_gender = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 2),
        )
        self.fc_age = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3),
        )
        # self.fc_mask = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(), nn.Linear(256, 3)
        # )
        # self.fc_gender = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(), nn.Linear(256, 2)
        # )
        # self.fc_age = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(), nn.Linear(256, 3)
        # )

    def forward(self, x):
        mask = self.resnet_mask(x)
        gender = self.resnet_gender(x)
        age = self.resnet_age(x)

        mask = self.fc_mask(mask)
        gender = self.fc_gender(gender)
        age = self.fc_age(age)
        return (mask, gender, age)


################from mission source 4.13 edited


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_num,
            out_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
    )


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(
            in_channels, reduced_channels, kernel_size=1, padding=0
        )
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            conv_batch(3, 32),
            conv_batch(32, 64, stride=2),
            self.make_layer(block, in_channels=64, num_blocks=1),
            conv_batch(64, 128, stride=2),
            self.make_layer(block, in_channels=128, num_blocks=2),
            conv_batch(128, 256, stride=2),
            self.make_layer(block, in_channels=256, num_blocks=8),
            conv_batch(256, 512, stride=2),
            self.make_layer(block, in_channels=512, num_blocks=8),
            conv_batch(512, 1024, stride=2),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.classifier(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)
