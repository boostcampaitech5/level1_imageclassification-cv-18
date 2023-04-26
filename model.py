import torch.nn as nn
import torch.nn.functional as F
import timm
from torchsummary import summary
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.init as init
import torch


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
        self.layer = []
        self.layer.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=(7 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(64))

        self.layer.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(128))
        self.layer.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(128))
        self.layer.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(128))
        self.layer.append(nn.MaxPool2d(2))

        self.layer.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(256))
        self.layer.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(256))
        self.layer.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(256))
        self.layer.append(nn.MaxPool2d(2))

        self.layer.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(512))
        self.layer.append(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(512))
        self.layer.append(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
            )
        )
        self.layer.append(nn.ReLU())
        self.layer.append(nn.BatchNorm2d(512))
        self.layer.append(nn.MaxPool2d(2))

        self.layer.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layer.append(nn.Flatten())
        self.layer.append(nn.Linear(512, num_classes))

        self.net = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.net(x)

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=True, num_classes=num_classes
        )
        for name, param in self.backbone.named_parameters():
            if name not in ("classifier.weight", "classifier.bias"):
                param.requires_grad = True
            else:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class EfficientNet4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=num_classes
        )
        for name, param in self.backbone.named_parameters():
            if name not in ("classifier.weight", "classifier.bias"):
                param.requires_grad = True
            else:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet101", pretrained=True, num_classes=num_classes
        )
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class resnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet152", pretrained=True, num_classes=num_classes
        )
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class densenet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=True, num_classes=num_classes
        )
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class facenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
        ).eval()
        layers = []
        layers.append(nn.Linear(512, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(64, num_classes))
        self.classifier = nn.Sequential(*layers)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                module.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        x = self.classifier
        return self.classifier(features)
