import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import numpy as np


def load_model(saved_model, num_classes, device, i):
    model_cls = getattr(import_module("model"), args.model)  # model argument로 model 지정
    model = model_cls(num_classes=num_classes)  # 뭐냐 이건

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_weight_path = os.path.join(saved_model, f"best{i}.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    outputs = []

    for i in range(1, 5):
        model = load_model(model_dir, num_classes, device, i).to(device)

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
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                mask_outs, gender_outs, age_outs = model(images)

                mask_outs = torch.nn.functional.softmax(mask_outs).cpu().numpy()
                gender_outs = torch.nn.functional.softmax(gender_outs).cpu().numpy()
                age_outs = torch.nn.functional.softmax(age_outs).cpu().numpy()

                pred = np.concatenate((mask_outs, gender_outs, age_outs), axis=1)
                preds.extend(pred)
        outputs.append(preds)
    outputs = np.array(outputs)
    temp = np.array(outputs[0])

    for i in outputs[1:]:
        temp += i
    temp /= 4  # k=4
    mask_preds = np.argmax(temp[:, :3], -1) * 6
    gender_preds = np.argmax(temp[:, 3:5], -1) * 3
    age_preds = np.argmax(temp[:, 5:], -1)
    temp = pd.DataFrame(temp)
    # preds = mask_preds*6 + gender_preds*3 + age_preds
    preds = np.add(mask_preds, np.add(gender_preds, age_preds))
    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output.csv")
    temp.to_csv(os.path.join(output_dir, f"soft_output.csv"), index=False)
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
        type=int,
        default=[224, 224],
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
