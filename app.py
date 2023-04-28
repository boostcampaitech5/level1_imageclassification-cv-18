import io
import os
from typing import Tuple

import albumentations as A
import numpy as np
import streamlit as st
import streamlit.config as stconfig
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image

from confirm_button_hack import cache_on_button_press
from model import AdvancedEfficientNetB3


def album_trans_img(image_bytes: bytes) -> torch.Tensor:
    album_transform = A.Compose(
        [
            A.CenterCrop(300, 300),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2(),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    array_image = np.array(image)

    return album_transform(image=array_image)["image"].unsqueeze(0)


@st.cache
def load_model():
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = AdvancedEfficientNetB3(mask_class=3, gender_class=2, age_class=3).to(
        device
    )
    model1.load_state_dict(torch.load(config["model_path1"], map_location=device))
    model2 = AdvancedEfficientNetB3(mask_class=3, gender_class=2, age_class=3).to(
        device
    )
    model2.load_state_dict(torch.load(config["model_path2"], map_location=device))
    model3 = AdvancedEfficientNetB3(mask_class=3, gender_class=2, age_class=3).to(
        device
    )
    model3.load_state_dict(torch.load(config["model_path3"], map_location=device))
    model4 = AdvancedEfficientNetB3(mask_class=3, gender_class=2, age_class=3).to(
        device
    )
    model4.load_state_dict(torch.load(config["model_path4"], map_location=device))

    return model1, model2, model3, model4


def get_prediction(models, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = album_trans_img(image_bytes=image_bytes).to(device)

    outputs = []
    with torch.no_grad():
        for model in models:
            out = model(image_tensor)

            (mask_out, gender_out, age_out) = torch.split(out, [3, 2, 3], dim=1)
            mask_out = F.softmax(mask_out).cpu().numpy()
            gender_out = F.softmax(gender_out).cpu().numpy()
            age_out = F.softmax(age_out).cpu().numpy()

            concat_out = np.concatenate((mask_out, gender_out, age_out), axis=1)
            outputs.append(concat_out)

    outputs = np.array(outputs)
    final_pred = outputs[0]
    for rest_output in outputs[1:]:
        final_pred += rest_output

    mask_pred = np.argmax(final_pred[:, :3], axis=-1) * 6
    gender_pred = np.argmax(final_pred[:, 3:5], axis=-1) * 3
    age_pred = np.argmax(final_pred[:, 5:], axis=-1)
    label = np.add(mask_pred, np.add(gender_pred, age_pred))

    return image_tensor, label


st.set_page_config(layout="wide")

def main():
    stconfig.set_option("client.showErrorDetails", False)
    st.title("Masked Face Image Classification Model")
    st.error("Advanced EfficientNet v1 B3 version was used")
    st.warning("K-Fold Cross Validation and Soft-Voting method were used")

    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model1, model2, model3, model4 = load_model()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    models = [model1, model2, model3, model4]

    uploaded_image = st.file_uploader("Choose An Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image")

        st.success(".....NOW CLASSIFYING.....")
        _, label = get_prediction(models=models, image_bytes=image_bytes)
        label = config["classes"][label.item()]
        st.info(f"Class Label Prediction : {label}")



main()