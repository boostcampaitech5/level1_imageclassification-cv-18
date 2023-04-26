import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import models
from tqdm import tqdm


# ! EfficientNet v1 - B3 Model Transfer Learning with Fine-Tuning Version
class AdvancedEfficientNetB3(nn.Module):
    def __init__(self, mask_class=3, gender_class=2, age_class=3):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True
        )  # TODO Input Size = 3 X 300 X 300
        self.feature_extractor = nn.Sequential(
            self.backbone.conv_stem,
            self.backbone.bn1,
            self.backbone.blocks,
            self.backbone.conv_head,
            self.backbone.bn2,
        )  # TODO Output Feature = 1536 X 10 X 10
        self.avg_pool = self.backbone.global_pool  # TODO Output Feature = 1536 X 1 X 1

        self.a_classifier = (
            nn.Sequential(  # ! ① A-Classifier for Mask Classification => 3
                nn.Flatten(),
                nn.Linear(in_features=1536, out_features=128, bias=True),
                nn.BatchNorm1d(128),
                nn.Hardswish(),
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=128, out_features=mask_class),
            )
        )
        self.b_classifier = (
            nn.Sequential(  # ! ② B-Classifier for Gender Classification => 2
                nn.Flatten(),
                nn.Linear(in_features=1536, out_features=128, bias=True),
                nn.BatchNorm1d(128),
                nn.Hardswish(),
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=128, out_features=gender_class),
            )
        )
        self.c_classifier = (
            nn.Sequential(  # ! ③ C-Classifier for Age Classification => 3
                nn.Flatten(),
                nn.Linear(in_features=1536, out_features=128, bias=True),
                nn.BatchNorm1d(128),
                nn.Hardswish(),
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=128, out_features=age_class),
            )
        )

        self._param_freeze(self.feature_extractor)
        self._param_freeze(self.avg_pool)

    def _param_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input):
        temp1 = self.feature_extractor(input)
        temp2 = self.avg_pool(temp1)

        temp_mask = self.a_classifier(temp2)
        temp_gender = self.b_classifier(temp2)
        temp_age = self.c_classifier(temp2)

        output = torch.cat(
            [temp_mask, temp_gender, temp_age], dim=1
        )  # TODO Output = [Mask-3, Gender-2, Age-3] Concat
        return output
