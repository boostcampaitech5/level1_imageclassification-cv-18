import argparse
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import wandb
from sklearn.metrics import f1_score


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
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
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


def train(data_dir, model_dir_mask, model_dir_gender, model_dir_age, args):
    seed_everything(args.seed)
    wandb.init(
        # set the wandb project where this run will be logged
        project="mask_classification",
        # track hyperparameters and run metadata
        config=vars(args),
    )
    save_dir_mask = increment_path(os.path.join(model_dir_mask, args.name))
    save_dir_gender = increment_path(os.path.join(model_dir_gender, args.name))
    save_dir_age = increment_path(os.path.join(model_dir_age, args.name))
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model_mask = model_module(num_classes=3).to(device)
    model_mask = torch.nn.DataParallel(model_mask)
    model_gender = model_module(num_classes=2).to(device)
    model_gender = torch.nn.DataParallel(model_gender)
    model_age = model_module(num_classes=3).to(device)
    model_age = torch.nn.DataParallel(model_age)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer_mask = opt_module(
        filter(lambda p: p.requires_grad, model_mask.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler_mask = StepLR(optimizer_mask, args.lr_decay_step, gamma=0.5)
    optimizer_gender = opt_module(
        filter(lambda p: p.requires_grad, model_gender.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler_gender = StepLR(optimizer_gender, args.lr_decay_step, gamma=0.5)
    optimizer_age = opt_module(
        filter(lambda p: p.requires_grad, model_age.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler_age = StepLR(optimizer_age, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir_mask)
    with open(os.path.join(save_dir_mask, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    task_names = ["mask", "gender", "age"]
    models = [model_mask, model_gender, model_age]
    optimizer = [optimizer_mask, optimizer_gender, optimizer_age]

    best_val_loss = np.inf
    best_val_mask_acc = 0
    best_val_mask_loss = np.inf
    best_val_gender_acc = 0
    best_val_gender_loss = np.inf
    best_val_age_acc = 0
    best_val_age_loss = np.inf

    for epoch in range(args.epochs):
        # train loop
        for model in models:
            model.train()
        loss_value = 0
        loss_value_mask = 0
        loss_value_gender = 0
        loss_value_age = 0
        matches = 0
        matches_mask = 0
        matches_gender = 0
        matches_age = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)

            optimizer_mask.zero_grad()
            optimizer_gender.zero_grad()
            optimizer_age.zero_grad()

            mask_label, gender_label, age_label = labels
            mask_label = mask_label.to(device)
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)
            labels = mask_label * 6 + gender_label * 3 + age_label

            mask_out, gender_out, age_out = (
                model_mask(inputs),
                model_gender(inputs),
                model_age(inputs),
            )
            mask_pred = torch.argmax(mask_out, dim=-1)
            gender_pred = torch.argmax(gender_out, dim=-1)
            age_pred = torch.argmax(age_out, dim=-1)
            preds = mask_pred * 6 + gender_pred * 3 + age_pred

            mask_loss = criterion(mask_out, mask_label)
            gender_loss = criterion(gender_out, gender_label)
            age_loss = criterion(age_out, age_label)
            loss = mask_loss + gender_loss + 1.5 * age_loss

            mask_loss.backward()
            optimizer_mask.step()
            gender_loss.backward()
            optimizer_gender.step()
            age_loss.backward()
            optimizer_age.step()

            loss_value += loss.item()
            loss_value_mask += mask_loss.item()
            loss_value_gender += gender_loss.item()
            loss_value_age += age_loss.item()
            matches += (preds == labels).sum().item()
            matches_mask += (mask_pred == mask_label).sum().item()
            matches_gender += (gender_pred == gender_label).sum().item()
            matches_age += (age_pred == age_label).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_loss_mask = loss_value_mask / args.log_interval
                train_loss_gender = loss_value_gender / args.log_interval
                train_loss_age = loss_value_age / args.log_interval

                train_acc = matches / args.batch_size / args.log_interval
                train_acc_mask = matches_mask / args.batch_size / args.log_interval
                train_acc_gender = matches_gender / args.batch_size / args.log_interval
                train_acc_age = matches_age / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer_mask)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss mask {train_loss_mask:4.4} || training accuracy mask {train_acc_mask:4.2%} || lr {current_lr}"
                )
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss gender {train_loss_gender:4.4} || training accuracy gender {train_acc_gender:4.2%} || lr {current_lr}"
                )
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss age {train_loss_age:4.4} || training accuracy age {train_acc_age:4.2%} || lr {current_lr}"
                )
                # logger.add_scalar(
                #     "Train/loss", train_loss, epoch * len(train_loader) + idx
                # )
                # logger.add_scalar(
                #     "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                # )
                wandb.log(
                    {
                        "Train/loss": train_loss,
                        "Train/accuracy": train_acc,
                        "Train/loss_mask": train_loss_mask,
                        "Train/accuracy_mask": train_acc_mask,
                        "Train/loss_gender": train_loss_gender,
                        "Train/accuracy_gender": train_acc_gender,
                        "Train/loss_age": train_loss_age,
                        "Train/accuracy_age": train_acc_age,
                    }
                )
                loss_value = 0
                loss_value_mask = 0
                loss_value_gender = 0
                loss_value_age = 0
                matches = 0
                matches_mask = 0
                matches_gender = 0
                matches_age = 0

        scheduler_mask.step()
        scheduler_gender.step()
        scheduler_age.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            for model in models:
                model.eval()
            val_loss_items = []
            val_mask_loss_items = []
            val_gender_loss_items = []
            val_age_loss_items = []
            val_acc_items = []
            val_f1_items = []
            val_mask_acc_items = []
            val_gender_acc_items = []
            val_age_acc_items = []
            val_mask_f1_items = []
            val_gender_f1_items = []
            val_age_f1_items = []

            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)

                mask_label, gender_label, age_label = labels
                mask_label = mask_label.to(device)
                gender_label = gender_label.to(device)
                age_label = age_label.to(device)
                labels = mask_label * 6 + gender_label * 3 + age_label

                mask_out, gender_out, age_out = (
                    model_mask(inputs),
                    model_gender(inputs),
                    model_age(inputs),
                )
                mask_pred = torch.argmax(mask_out, dim=-1)
                gender_pred = torch.argmax(gender_out, dim=-1)
                age_pred = torch.argmax(age_out, dim=-1)
                preds = mask_pred * 6 + gender_pred * 3 + age_pred

                mask_loss = criterion(mask_out, mask_label)
                gender_loss = criterion(gender_out, gender_label)
                age_loss = criterion(age_out, age_label)
                loss = mask_loss + gender_loss + 1.5 * age_loss
                loss_item = loss.item()

                acc_item = (labels == preds).sum().item()
                mask_acc_item = (mask_label == mask_pred).sum().item()
                gender_acc_item = (gender_label == gender_pred).sum().item()
                age_acc_item = (age_label == age_pred).sum().item()

                f1_item = f1_score(
                    torch.clone(labels).detach().cpu(),
                    preds.cpu(),
                    average="weighted",
                ).item()
                mask_f1_item = f1_score(
                    torch.clone(mask_label).detach().cpu(),
                    mask_pred.cpu(),
                    average="weighted",
                ).item()
                gender_f1_item = f1_score(
                    torch.clone(gender_label).detach().cpu(),
                    gender_pred.cpu(),
                    average="weighted",
                ).item()
                age_f1_item = f1_score(
                    torch.clone(age_label).detach().cpu(),
                    age_pred.cpu(),
                    average="weighted",
                ).item()

                val_loss_items.append(loss_item)
                val_mask_loss_items.append(mask_loss.item())
                val_gender_loss_items.append(gender_loss.item())
                val_age_loss_items.append(age_loss.item())

                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

                val_mask_f1_items.append(mask_f1_item)
                val_gender_f1_items.append(gender_f1_item)
                val_age_f1_items.append(age_f1_item)

                val_mask_acc_items.append(mask_acc_item)
                val_gender_acc_items.append(gender_acc_item)
                val_age_acc_items.append(age_acc_item)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
            val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
            val_age_loss = np.sum(val_age_loss_items) / len(val_loader)

            val_acc = np.sum(val_acc_items) / len(val_set)
            val_mask_acc = np.sum(val_mask_acc_items) / len(val_set)
            val_gender_acc = np.sum(val_gender_acc_items) / len(val_set)
            val_age_acc = np.sum(val_age_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            val_mask_f1 = np.sum(val_mask_f1_items) / len(val_loader)
            val_gender_f1 = np.sum(val_gender_f1_items) / len(val_loader)
            val_age_f1 = np.sum(val_age_f1_items) / len(val_loader)

            best_val_loss = min(best_val_loss, val_loss)
            best_val_mask_loss = min(best_val_mask_loss, val_mask_loss)
            best_val_gender_loss = min(best_val_gender_loss, val_gender_loss)
            best_val_age_loss = min(best_val_age_loss, val_age_loss)

            if val_mask_acc > best_val_mask_acc:
                print(
                    f"New best model for val mask accuracy : {val_mask_acc:4.2%}! saving the best model.."
                )
                torch.save(
                    model_mask.module.state_dict(), f"{save_dir_mask}/best_mask.pth"
                )
                best_val_mask_acc = val_mask_acc
            torch.save(model_mask.module.state_dict(), f"{save_dir_mask}/last_mask.pth")
            print(
                f"[Val] mask acc : {val_mask_acc:4.2%}, mask loss: {val_mask_loss:4.2} || "
                f"best mask acc : {best_val_mask_acc:4.2%}, best mask loss: {best_val_mask_loss:4.2}"
            )
            if val_gender_acc > best_val_gender_acc:
                print(
                    f"New best model for val gender accuracy : {val_gender_acc:4.2%}! saving the best model.."
                )
                torch.save(
                    model_gender.module.state_dict(),
                    f"{save_dir_gender}/best_gender.pth",
                )
                best_val_gender_acc = val_gender_acc
            torch.save(
                model_gender.module.state_dict(), f"{save_dir_gender}/last_gender.pth"
            )
            print(
                f"[Val] gender acc : {val_gender_acc:4.2%}, gender loss: {val_gender_loss:4.2} || "
                f"best gender acc : {best_val_gender_acc:4.2%}, best gender loss: {best_val_gender_loss:4.2}"
            )
            if val_age_acc > best_val_age_acc:
                print(
                    f"New best model for val age accuracy : {val_age_acc:4.2%}! saving the best model.."
                )
                torch.save(
                    model_age.module.state_dict(), f"{save_dir_age}/best_age.pth"
                )
                best_val_age_acc = val_age_acc
            torch.save(model_age.module.state_dict(), f"{save_dir_age}/last_age.pth")
            print(
                f"[Val] age acc : {val_age_acc:4.2%}, age loss: {val_age_loss:4.2} || "
                f"best age acc : {best_val_age_acc:4.2%}, best age loss: {best_val_age_loss:4.2}"
            )
            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            wandb.log(
                {
                    "Val/loss": val_loss,
                    "Val/accuracy": val_acc,
                    "Val/f1_score": val_f1,
                    "Val/mask_acc": val_mask_acc,
                    "Val/gender_acc": val_gender_acc,
                    "Val/age_acc": val_age_acc,
                    "Val/mask_f1": val_mask_f1,
                    "Val/gender_f1": val_gender_f1,
                    "Val/age_f1": val_age_f1,
                    "Val/mask_loss": val_mask_loss,
                    "Val/gender_loss": val_gender_loss,
                    "Val/age_loss": val_age_loss,
                }
            )
            print()
    # logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=int,
        default=[224, 224],
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
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)"
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
        default=20,
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
        "--model_dir_mask", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )
    parser.add_argument(
        "--model_dir_gender",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./model"),
    )
    parser.add_argument(
        "--model_dir_age", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir_mask = args.model_dir_mask
    model_dir_gender = args.model_dir_gender
    model_dir_age = args.model_dir_age

    train(data_dir, model_dir_mask, model_dir_gender, model_dir_age, args)
