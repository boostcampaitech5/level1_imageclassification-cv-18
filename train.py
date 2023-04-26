import argparse
import copy
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import AdvancedMaskDataset, BaseMaskDataset, apply_transforms
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # ! cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # ! cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = BaseMaskDataset.decode_multi_class(gt)
        pred_decoded_labels = BaseMaskDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    torch.cuda.empty_cache()

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(  # ! AdvancedMaskDataset
        import_module("dataset"), args.dataset
    )
    dataset1 = dataset_module(
        data_dir=data_dir,
    )
    mean, std = dataset1.mean, dataset1.std
    album_transform = apply_transforms(
        mean=mean, std=std
    )  # ! Albumentation Augmentations with Train and Valid

    kfold = KFold(n_splits=4, shuffle=True, random_state=2023)
    all_index = 0
    for train_idx, valid_idx in kfold.split(dataset1):
        dataset2 = copy.deepcopy(dataset1)
        train_set = Subset(dataset1, train_idx)
        valid_set = Subset(dataset2, valid_idx)

        train_set.dataset.set_transform(album_transform["train"])
        valid_set.dataset.set_transform(album_transform["valid"])

        all_index += 1

        # -- data_loader
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)
        model = model_module(mask_class=3, gender_class=2, age_class=3).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion1 = create_criterion(args.criterion)
        criterion2 = torch.nn.CrossEntropyLoss()
        criterion3 = create_criterion(args.criterion)
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_valid_acc = 0
        best_valid_loss = np.inf
        for epoch in tqdm(range(args.epochs), desc=f"{all_index}-Training..."):
            model.train()  # ! Training Loop

            loss_value = 0
            mask_matches, gender_matches, age_matches, matches = 0, 0, 0, 0
            train_f1_score = 0

            for idx, train_batch in enumerate(train_loader):
                inputs, (mask_labels, gender_labels, age_labels) = train_batch

                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                labels = mask_labels * 6 + gender_labels * 3 + age_labels

                optimizer.zero_grad()

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
                mask_loss = criterion1(mask_outs, mask_labels)
                gender_loss = criterion2(gender_outs, gender_labels)
                age_loss = criterion3(age_outs, age_labels)
                loss = mask_loss + gender_loss + age_loss

                loss.backward()
                optimizer.step()

                loss_value += loss.item()

                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)
                preds = mask_preds * 6 + gender_preds * 3 + age_preds

                mask_matches += torch.sum(mask_preds == mask_labels).item()
                gender_matches += torch.sum(gender_preds == gender_labels).item()
                age_matches += torch.sum(age_preds == age_labels).item()
                matches += torch.sum(preds == labels).item()

                train_f1_score += f1_score(
                    preds.cpu().numpy(), labels.cpu().numpy(), average="macro"
                )

                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    train_f1_score = train_f1_score / args.log_interval

                    train_mask_acc = mask_matches / args.batch_size / args.log_interval
                    train_gender_acc = (
                        gender_matches / args.batch_size / args.log_interval
                    )
                    train_age_acc = age_matches / args.batch_size / args.log_interval

                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || LR={current_lr} || "
                        f"Train Loss={train_loss:<4.4} || Train Acc={train_acc:<4.2%} || F1 Score={train_f1_score:<4.4}"
                    )
                    print(
                        f"[Train] Mask Acc={train_mask_acc:<4.2%} || Gender Acc={train_gender_acc:<4.2%} || Age Acc={train_age_acc:<4.2%}"
                    )
                    logger.add_scalar(
                        "Train/Loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/Acc", train_acc, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/F1", train_f1_score, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    mask_matches = 0
                    gender_matches = 0
                    age_matches = 0
                    matches = 0
                    train_f1_score = 0

            with torch.no_grad():  # ! Validation Loop
                print("Calculating validation results...")
                model.eval()

                valid_loss_items = []
                valid_acc_items = []
                mask_matches, gender_matches, age_matches = 0, 0, 0
                valid_f1_score = 0
                figure = None

                for valid_batch in valid_loader:
                    inputs, (mask_labels, gender_labels, age_labels) = valid_batch

                    inputs = inputs.to(device)
                    mask_labels = mask_labels.to(device)
                    gender_labels = gender_labels.to(device)
                    age_labels = age_labels.to(device)

                    labels = mask_labels * 6 + gender_labels * 3 + age_labels

                    outs = model(inputs)
                    (mask_outs, gender_outs, age_outs) = torch.split(
                        outs, [3, 2, 3], dim=1
                    )

                    mask_loss_item = criterion1(mask_outs, mask_labels).item()
                    gender_loss_item = criterion2(gender_outs, gender_labels).item()
                    age_loss_item = criterion3(age_outs, age_labels).item()
                    loss_item = mask_loss_item + gender_loss_item + age_loss_item
                    valid_loss_items.append(loss_item)

                    mask_preds = torch.argmax(mask_outs, dim=-1)
                    gender_preds = torch.argmax(gender_outs, dim=-1)
                    age_preds = torch.argmax(age_outs, dim=-1)
                    preds = mask_preds * 6 + gender_preds * 3 + age_preds

                    acc_item = (labels == preds).sum().item()
                    valid_acc_items.append(acc_item)

                    mask_matches += torch.sum(mask_preds == mask_labels).item()
                    gender_matches += torch.sum(gender_preds == gender_labels).item()
                    age_matches += torch.sum(age_preds == age_labels).item()

                    valid_f1_score += f1_score(
                        preds.cpu().numpy(), labels.cpu().numpy(), average="macro"
                    )

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs)
                            .detach()
                            .cpu()
                            .permute(0, 2, 3, 1)
                            .numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset2.mean, dataset2.std
                        )
                        figure = grid_image(
                            inputs_np,
                            labels,
                            preds,
                            n=16,
                            shuffle=args.dataset != "AdvancedMaskDataset",
                        )

                valid_loss = np.sum(valid_loss_items) / len(valid_loader)
                valid_acc = np.sum(valid_acc_items) / len(valid_set)
                valid_mask_acc = mask_matches / len(valid_set)
                valid_gender_acc = gender_matches / len(valid_set)
                valid_age_acc = age_matches / len(valid_set)

                valid_f1_score = valid_f1_score / len(valid_loader)

                best_valid_loss = min(best_valid_loss, valid_loss)
                if valid_acc > best_valid_acc:
                    print(
                        f"New best model for Valid Acc = {valid_acc:<4.2%}! saving the best model.."
                    )
                    torch.save(
                        model.module.state_dict(),
                        f"{save_dir}/best_{all_index}fold.pth",
                    )
                    best_valid_acc = valid_acc

                torch.save(
                    model.module.state_dict(), f"{save_dir}/last_{all_index}fold.pth"
                )
                print(
                    f"Valid Loss={valid_loss:<4.2} || Valid Acc={valid_acc:<4.2%} || F1 Score={valid_f1_score:<4.4}"
                )
                print(
                    f"[Valid] Mask Acc={valid_mask_acc:<4.2%} || Gender Acc={valid_gender_acc:<4.2%} || Age Acc={valid_age_acc:<4.2%}"
                )
                print(
                    f"Best Acc={best_valid_acc:<4.2%} & Best Loss={best_valid_loss:<4.2}"
                )
                logger.add_scalar("Valid/Loss", valid_loss, epoch)
                logger.add_scalar("Valid/Acc", valid_acc, epoch)
                logger.add_scalar("Valid/F1", valid_f1_score, epoch)
                logger.add_figure("Results", figure, epoch)
                print()
                print()

                scheduler.step(valid_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=12, help="random seed (default: 12)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AdvancedMaskDataset",
        help="dataset augmentation type (default: AdvancedMaskDataset)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=list,
        default=[300, 300],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=500,
        help="input batch size for validing (default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="AdvancedEfficientNetB3",
        help="model type (default: AdvancedEfficientNetB3)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
