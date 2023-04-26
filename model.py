import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
import torchvision


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
class DarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.darknet = timm.create_model("darknet53", num_classes=0)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, self.num_classes, bias=True)
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        net = self.darknet.forward_features(x)
        net = self.classifier(net)

        return net


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.backbone = torchvision.models.resnet18(pretrained=True)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(512, 256), torch.nn.Linear(256, num_classes)
        # )

        self.feature_extract = timm.create_model(
            "resnet50", pretrained=True, num_classes=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes, bias=True),
        )

    def forward(self, input):
        net = self.feature_extract(input)
        net = self.classifier(net)
        return net


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.backbone = torchvision.models.resnet18(pretrained=True)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(512, 256), torch.nn.Linear(256, num_classes)
        # )

        self.feature_extract = timm.create_model(
            "resnet18", pretrained=True, num_classes=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
        )

    def forward(self, input):
        net = self.feature_extract(input)
        net = self.classifier(net)
        return net


if __name__ == "__main__":
    from torchsummary import summary

    images = torch.rand([3, 3, 384, 512]).float()

    model = ResNet18(18)
    # summary(model, (3, 256, 256), device="cpu")

    print(model(images).shape)
