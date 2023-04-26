import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset


def load_model(saved_model, device, index):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(mask_class=3, gender_class=2, age_class=3)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f"best_{index}fold.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    infer_outputs = []

    for all_index in range(1, 5):
        model = load_model(model_dir, device, all_index).to(device)
        model.eval()

        img_root = os.path.join(data_dir, "images")
        info_path = os.path.join(data_dir, "info.csv")
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 4,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        model_outs = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                outs = model(images)

                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
                mask_outs = F.softmax(mask_outs).cpu().numpy()
                gender_outs = F.softmax(gender_outs).cpu().numpy()
                age_outs = F.softmax(age_outs).cpu().numpy()

                outs = np.concatenate((mask_outs, gender_outs, age_outs), axis=1)
                model_outs.extend(outs)
        infer_outputs.append(model_outs)
    infer_outputs = np.array(infer_outputs)

    temp_outputs = np.array(infer_outputs[0])
    for next_model_out in infer_outputs[1:]:
        temp_outputs += next_model_out
    temp_outputs /= 4

    soft_voting = pd.DataFrame(temp_outputs, columns=range(8))
    save_path = os.path.join(output_dir, f"kjj_voting.csv")
    soft_voting.to_csv(save_path, index=False)

    mask_preds = np.argmax(temp_outputs[:, :3], axis=-1) * 6
    gender_preds = np.argmax(temp_outputs[:, 3:5], axis=-1) * 3
    age_preds = np.argmax(temp_outputs[:, 5:], axis=-1)
    preds = np.add(mask_preds, np.add(gender_preds, age_preds))

    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="input batch size for validing (default: 200)",
    )
    parser.add_argument(
        "--resize",
        type=tuple,
        default=(300, 300),
        help="resize size for image when you trained (default: (300, 300))",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="AdvancedEfficientNetB3",
        help="model type (default: AdvancedEfficientNetB3)",
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
