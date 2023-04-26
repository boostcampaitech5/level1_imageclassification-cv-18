import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet_b0, self).__init__()

        self.survived_numerator = 80
        self.survived_denominator = 80

        layers = []
        layers.append(
            self.ConvBlock(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=4,
                    survived_ratio=self.get_survived_ratio(),
                )
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=16,
                    out_channels=24,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=24,
                    out_channels=40,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=40,
                    out_channels=40,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=40,
                    out_channels=80,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=80,
                    out_channels=80,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=80,
                    out_channels=80,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=80,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=112,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=112,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=112,
                    out_channels=192,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=192,
                    out_channels=320,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                )
            )
        )
        layers.append(self.ConvBlock(in_channels=320, out_channels=1280, kernel_size=1))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512, bias=True),
            nn.Linear(512, num_classes, bias=True),
        )

    def get_survived_ratio(self):
        ratio = self.survived_numerator / self.survived_denominator
        self.survived_numerator -= 1
        return ratio

    def forward(self, x):
        out = self.features(x)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    class MBConv(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            se_ratio,
            survived_ratio,
            expand_ratio=1,
        ):
            super(EfficientNet_b0.MBConv, self).__init__()

            self.survived_ratio = survived_ratio

            self.is_skip_connection = in_channels == out_channels

            layers = []

            if expand_ratio != 1:
                layers.append(
                    EfficientNet_b0.ConvBlock(
                        in_channels=in_channels,
                        out_channels=in_channels * expand_ratio,
                        kernel_size=1,
                        stride=1,
                    )
                )
                in_channels *= expand_ratio
            layers.append(
                EfficientNet_b0.ConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                )
            )
            layers.append(self.SE_Layer(in_channels=in_channels, se_ratio=se_ratio))
            layers.append(
                EfficientNet_b0.ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    act_func=False,
                )
            )
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            inputs = x

            out = self.block(x)

            # Stochastic Dropout & Short Cut Connection
            if self.training and 0 < self.survived_ratio <= 1:
                ratio = self.survived_ratio
                ratio += torch.rand(
                    [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device
                )
                binary = torch.floor(ratio)
                out = out / self.survived_ratio * binary

            if self.is_skip_connection:
                out += inputs

            return out

        class SE_Layer(nn.Module):
            def __init__(self, in_channels, se_ratio):
                super(EfficientNet_b0.MBConv.SE_Layer, self).__init__()
                hidden_channels = in_channels // se_ratio

                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
                )
                self.fc2 = nn.Conv2d(
                    in_channels=hidden_channels, out_channels=in_channels, kernel_size=1
                )
                self.activation = nn.SiLU(inplace=True)
                self.scale_activation = nn.Sigmoid()

            def forward(self, x):
                out = self.avgpool(x)
                out = self.fc1(out)
                out = self.activation(out)
                out = self.fc2(out)
                out = self.scale_activation(out)
                return out * x

    class ConvBlock(nn.Sequential):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            act_func=True,
        ):
            layers = []

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            if act_func:
                layers.append(nn.SiLU(inplace=True))
            super().__init__(*layers)


class EfficientNet_b1(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet_b1, self).__init__()

        self.survived_numerator = 115
        self.survived_denominator = 115

        layers = []
        layers.append(
            self.ConvBlock(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=4,
                    survived_ratio=self.get_survived_ratio(),
                ),
                self.MBConv(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=4,
                    survived_ratio=self.get_survived_ratio(),
                ),
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=16,
                    out_channels=24,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=24,
                    out_channels=40,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=40,
                    out_channels=40,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=40,
                    out_channels=40,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=40,
                    out_channels=80,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=80,
                    out_channels=80,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=80,
                    out_channels=80,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=80,
                    out_channels=80,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=80,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=112,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=112,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=112,
                    out_channels=112,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=112,
                    out_channels=192,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=192,
                    out_channels=192,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=192,
                    out_channels=320,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=320,
                    out_channels=320,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(self.ConvBlock(320, 1280, kernel_size=1))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512, bias=True),
            nn.Linear(512, num_classes, bias=True),
        )

    def get_survived_ratio(self):
        ratio = self.survived_numerator / self.survived_denominator
        self.survived_numerator -= 1
        return ratio

    def forward(self, x):
        out = self.features(x)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    class MBConv(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            se_ratio,
            survived_ratio,
            expand_ratio=1,
        ):
            super(EfficientNet_b1.MBConv, self).__init__()

            self.survived_ratio = survived_ratio

            self.is_skip_connection = in_channels == out_channels

            layers = []

            if expand_ratio != 1:
                layers.append(
                    EfficientNet_b1.ConvBlock(
                        in_channels=in_channels,
                        out_channels=in_channels * expand_ratio,
                        kernel_size=1,
                        stride=1,
                    )
                )
                in_channels *= expand_ratio
            layers.append(
                EfficientNet_b1.ConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                )
            )
            layers.append(self.SE_Layer(in_channels=in_channels, se_ratio=se_ratio))
            layers.append(
                EfficientNet_b1.ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    act_func=False,
                )
            )
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            inputs = x

            out = self.block(x)

            # Stochastic Dropout & Short Cut Connection
            if self.training and 0 < self.survived_ratio <= 1:
                ratio = self.survived_ratio
                ratio += torch.rand(
                    [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device
                )
                binary = torch.floor(ratio)
                out = out / self.survived_ratio * binary

            if self.is_skip_connection:
                out += inputs

            return out

        class SE_Layer(nn.Module):
            def __init__(self, in_channels, se_ratio):
                super(EfficientNet_b1.MBConv.SE_Layer, self).__init__()
                hidden_channels = in_channels // se_ratio

                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
                )
                self.fc2 = nn.Conv2d(
                    in_channels=hidden_channels, out_channels=in_channels, kernel_size=1
                )
                self.activation = nn.SiLU(inplace=True)
                self.scale_activation = nn.Sigmoid()

            def forward(self, x):
                out = self.avgpool(x)
                out = self.fc1(out)
                out = self.activation(out)
                out = self.fc2(out)
                out = self.scale_activation(out)
                return out * x

    class ConvBlock(nn.Sequential):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            act_func=True,
        ):
            layers = []

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            if act_func:
                layers.append(nn.SiLU(inplace=True))
            super().__init__(*layers)


class EfficientNet_b2(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet_b2, self).__init__()

        self.survived_numerator = 115
        self.survived_denominator = 115

        layers = []
        layers.append(
            self.ConvBlock(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=4,
                    survived_ratio=self.get_survived_ratio(),
                ),
                self.MBConv(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=4,
                    survived_ratio=self.get_survived_ratio(),
                ),
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=16,
                    out_channels=24,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )

        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=24,
                    out_channels=48,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=48,
                    out_channels=88,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=88,
                    out_channels=88,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=88,
                    out_channels=88,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=88,
                    out_channels=88,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=88,
                    out_channels=120,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=120,
                    out_channels=120,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=120,
                    out_channels=120,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=120,
                    out_channels=120,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=120,
                    out_channels=208,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=208,
                    out_channels=208,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=208,
                    out_channels=208,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=208,
                    out_channels=208,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=208,
                    out_channels=208,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(
            nn.Sequential(
                self.MBConv(
                    in_channels=208,
                    out_channels=352,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
                self.MBConv(
                    in_channels=352,
                    out_channels=352,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    se_ratio=24,
                    survived_ratio=self.get_survived_ratio(),
                    expand_ratio=6,
                ),
            )
        )
        layers.append(self.ConvBlock(in_channels=352, out_channels=1408, kernel_size=1))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, 1024, bias=True),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, num_classes, bias=True),
        )

    def get_survived_ratio(self):
        ratio = self.survived_numerator / self.survived_denominator
        self.survived_numerator -= 1
        return ratio

    def forward(self, x):
        out = self.features(x)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    class MBConv(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            se_ratio,
            survived_ratio,
            expand_ratio=1,
        ):
            super(EfficientNet_b2.MBConv, self).__init__()

            self.survived_ratio = survived_ratio

            self.is_skip_connection = in_channels == out_channels

            layers = []

            if expand_ratio != 1:
                layers.append(
                    EfficientNet_b2.ConvBlock(
                        in_channels=in_channels,
                        out_channels=in_channels * expand_ratio,
                        kernel_size=1,
                        stride=1,
                    )
                )
                in_channels *= expand_ratio
            layers.append(
                EfficientNet_b2.ConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                )
            )
            layers.append(self.SE_Layer(in_channels=in_channels, se_ratio=se_ratio))
            layers.append(
                EfficientNet_b2.ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    act_func=False,
                )
            )
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            inputs = x

            out = self.block(x)

            # Stochastic Dropout & Short Cut Connection
            if self.training and 0 < self.survived_ratio <= 1:
                ratio = self.survived_ratio
                ratio += torch.rand(
                    [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device
                )
                binary = torch.floor(ratio)
                out = out / self.survived_ratio * binary

            if self.is_skip_connection:
                out += inputs

            return out

        class SE_Layer(nn.Module):
            def __init__(self, in_channels, se_ratio):
                super(EfficientNet_b2.MBConv.SE_Layer, self).__init__()
                hidden_channels = in_channels // se_ratio

                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
                )
                self.fc2 = nn.Conv2d(
                    in_channels=hidden_channels, out_channels=in_channels, kernel_size=1
                )
                self.activation = nn.SiLU(inplace=True)
                self.scale_activation = nn.Sigmoid()

            def forward(self, x):
                out = self.avgpool(x)
                out = self.fc1(out)
                out = self.activation(out)
                out = self.fc2(out)
                out = self.scale_activation(out)
                return out * x

    class ConvBlock(nn.Sequential):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            act_func=True,
        ):
            layers = []

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            if act_func:
                layers.append(nn.SiLU(inplace=True))
            super().__init__(*layers)


class Assemble(nn.Module):
    def __init__(self, num_classes) -> None:
        super(Assemble, self).__init__()
        self.num_classes = num_classes

        self.model0 = EfficientNet_b0(num_classes=self.num_classes)
        self.model0.features.load_state_dict(
            torch.load("/opt/ml/level1_imageclassification-cv-18/b0_pre_weights.pth")
        )

        self.model1 = EfficientNet_b1(num_classes=self.num_classes)
        self.model1.features.load_state_dict(
            torch.load("/opt/ml/level1_imageclassification-cv-18/b1_pre_weights.pth")
        )

        self.model2 = EfficientNet_b2(num_classes=self.num_classes)
        self.model2.features.load_state_dict(
            torch.load("/opt/ml/level1_imageclassification-cv-18/b2_pre_weights.pth")
        )

    def forward(self, x):
        predicts = []
        predicts.append(self.model0(x))
        predicts.append(self.model1(x))
        predicts.append(self.model2(x))

        predicts = torch.stack(predicts, dim=1)
        res = torch.mode(predicts, dim=1).values.view(-1, self.num_classes)

        return torch.Tensor(res)


class Timm_EfficientNet_b0(nn.Module):
    def __init__(self, num_classes):
        super(Timm_EfficientNet_b0, self).__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier_age = self.Classifier_age()
        self.classifier_sex = self.Classifier_sex()
        self.classifier_mask = self.Classifier_mask()

    def forward(self, x):
        features = self.backbone.forward_features(x)
        avg_pool = self.avgpool(features)
        age = self.classifier_age(avg_pool)
        sex = self.classifier_sex(avg_pool)
        mask = self.classifier_mask(avg_pool)

        output = torch.cat([mask, sex, age], dim=1)
        return output

    class Classifier_age(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 3, bias=True),
            )

    class Classifier_sex(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 2, bias=True),
            )

    class Classifier_mask(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 3, bias=True),
            )


class Ensemble(nn.Module):
    def __init__(self, num_classes) -> None:
        super(Ensemble, self).__init__()
        self.num_classes = num_classes
        self.out_classes = 8

        self.model0 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model1 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model2 = Timm_EfficientNet_b0(num_classes=self.out_classes)

    def forward(self, x):
        predicts = []
        predicts.append(self.model0(x))
        predicts.append(self.model1(x))
        predicts.append(self.model2(x))

        predicts = torch.stack(predicts, dim=1)
        res = torch.mode(predicts, dim=1).values.view(-1, self.out_classes)

        return res


class Ensemble5(nn.Module):
    def __init__(self, num_classes) -> None:
        super(Ensemble5, self).__init__()
        self.num_classes = num_classes
        self.out_classes = 8

        self.model0 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model1 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model2 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model3 = Timm_EfficientNet_b0(num_classes=self.out_classes)
        self.model4 = Timm_EfficientNet_b0(num_classes=self.out_classes)

    def forward(self, x):
        predicts = []
        predicts.append(self.model0(x))
        predicts.append(self.model1(x))
        predicts.append(self.model2(x))
        predicts.append(self.model3(x))
        predicts.append(self.model4(x))

        predicts = torch.stack(predicts, dim=1)
        res = torch.mode(predicts, dim=1).values.view(-1, self.out_classes)

        return res
