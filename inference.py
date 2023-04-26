import argparse
import multiprocessing
import os
from importlib import import_module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    resize = int(args.resize[0])

    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device).float()

            images_crop = [
                transforms.functional.crop(images, 0, 0, resize, resize),
                transforms.functional.crop(
                    images,
                    0,
                    images.shape[2] - resize,
                    resize,
                    resize,
                ),
                transforms.functional.crop(
                    images,
                    images.shape[2] - resize,
                    0,
                    resize,
                    resize,
                ),
                transforms.functional.crop(
                    images,
                    images.shape[2] - resize,
                    images.shape[2] - resize,
                    resize,
                    resize,
                ),
                transforms.functional.center_crop(images, resize),
            ]

            images_horizontal = [
                transforms.RandomHorizontalFlip(p=1.0)(i) for i in images_crop
            ]

            pred_crop = [model(i) for i in images_crop]
            pred_horizontal = [model(i) for i in images_horizontal]
            pred_crop = torch.stack(pred_crop, dim=1)
            pred_horizontal = torch.stack(pred_horizontal, dim=1)

            # batch, 10, 8
            pred = torch.cat([pred_crop, pred_horizontal], dim=1)
            # batch,
            (mask_outs, gender_outs, age_outs) = torch.split(pred, [3, 2, 3], dim=-1)

            # batch, 3
            mask_outs = torch.nn.functional.softmax(mask_outs, dim=-1).cpu().numpy()
            mask_outs = mask_outs.sum(axis=1) / 10.0

            # batch, 2
            gender_outs = torch.nn.functional.softmax(gender_outs, dim=-1).cpu().numpy()
            gender_outs = gender_outs.sum(axis=1) / 10.0

            # batch, 3
            age_outs = torch.nn.functional.softmax(age_outs, dim=-1).cpu().numpy()
            age_outs = age_outs.sum(axis=1) / 10.0

            mask_preds = mask_outs.argmax(-1)
            gender_preds = gender_outs.argmax(-1)
            age_preds = age_outs.argmax(-1)

            pred = mask_preds * 6 + gender_preds * 3 + age_preds
            preds.extend(pred)

    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output_sharpen_18_58.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )

    parser.add_argument(
        "--resize",
        type=tuple,
        default=(224, 224),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
