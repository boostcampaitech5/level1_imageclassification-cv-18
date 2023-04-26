import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from dataset import MaskBaseDataset
from torch.utils.data import Subset, Dataset
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
    assert n <= batch_size  # batch_size는 16이상이여야함

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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    print(save_dir)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # dataset에 접근해서 args.dataset으로 넣은 데이터셋 클래스를 가져오는 듯
    dataset = dataset_module(  # 즉 이건 dataset = BaseDataset(...)과 같다
        data_dir=data_dir,
    )
    num_classes = (
        dataset.num_classes
    )  # 18                           # for multilabel have to be "8" 심지어 datset안에 구현되있어서 여기서 클래스 선언하고 변경시켜야함...

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    val_transform_module = getattr(
        import_module("dataset"), "BaseAugmentation"
    )  # default: BaseAugmentation
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    kfold = KFold(n_splits=3, shuffle=True, random_state=2023)
    index = 0
    for train_index, val_index in kfold.split(dataset):
        dataset_ = deepcopy(dataset)
        train_set = Subset(dataset, train_index)
        val_set = Subset(dataset_, val_index)

        train_set.dataset.set_transform(transform)
        val_set.dataset.set_transform(val_transform)
        index += 1

        # train_set, val_set = dataset.split_dataset()
        # train_set.dataset.set_transform(transform)
        # val_set.dataset.set_transform(val_transform)
        # dataset.set_transform(transform)        #아래 train과 val쪼개주는데에서 다시 해줘야함

        # -- data_loader

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
        model = model_module(num_classes=num_classes).to(device)
        # model.init_param()

        model = torch.nn.DataParallel(model)  # to(device)이후 멀티 GPU사용가능하게함!

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(
            import_module("torch.optim"), args.optimizer
        )  # default: SGD     얘는 adam을 넣어보자
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4,
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)  # 검색한다음 사용할 것

        # -- logging
        logger = SummaryWriter(
            log_dir=save_dir
        )  # save_dir = ./model/args.name 폴더 생성을 얘가 해주는거임
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        figure = None
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            mask_matches = 0
            gender_matches = 0
            age_matches = 0
            matches = 0
            f1_train = 0
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

                mask_loss = criterion(mask_outs, mask_labels)
                gender_loss = criterion(gender_outs, gender_labels)
                age_loss = criterion(age_outs, age_labels)

                loss = mask_loss + gender_loss + age_loss
                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                mask_preds = mask_outs.argmax(1)
                # print(mask_preds)
                # print(torch.argmax(torch.nn.functional.softmax(mask_outs).detach().cpu(), dim=1))  ##확인결과 둘이 결과 똑같음 이걸로 soft voting하면 될듯?
                gender_preds = gender_outs.argmax(1)
                age_preds = age_outs.argmax(1)
                preds = (
                    mask_preds * 6 + gender_preds * 3 + age_preds
                )  # shape is [batch, 1]

                mask_matches += torch.sum(mask_preds == mask_labels).item()
                gender_matches += torch.sum(gender_preds == gender_labels).item()
                age_matches += torch.sum(age_preds == age_labels).item()
                matches += torch.sum(preds == labels).item()
                f1_train += f1_score(
                    preds.cpu().numpy(), labels.cpu().numpy(), average="macro"
                )

                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    # train_acc = matches / args.batch_size / args.log_interval
                    mask_acc = mask_matches / args.batch_size / args.log_interval
                    gender_acc = gender_matches / args.batch_size / args.log_interval
                    age_acc = age_matches / args.batch_size / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    f1_train = f1_train / args.log_interval

                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4}  || lr {current_lr} || F1 score {f1_train:4.4}"
                    )
                    print(
                        f"[Training] Mask acc: {mask_acc:4.2%}, Gender acc: {gender_acc:4.2%}, Age acc: {age_acc:4.2%}, Acc: {train_acc:4.2%}"
                    )
                    logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/f1", f1_train, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    mask_matches = 0
                    gender_matches = 0
                    age_matches = 0
                    matches = 0
                    f1_train = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                mask_matches = 0
                gender_matches = 0
                age_matches = 0
                matches = 0
                f1_test = 0
                for val_batch in val_loader:
                    inputs, (mask_labels, gender_labels, age_labels) = val_batch
                    inputs = inputs.to(device)
                    mask_labels = mask_labels.to(device)
                    gender_labels = gender_labels.to(device)
                    age_labels = age_labels.to(device)
                    labels = mask_labels * 6 + gender_labels * 3 + age_labels

                    outs = model(inputs)
                    (mask_outs, gender_outs, age_outs) = torch.split(
                        outs, [3, 2, 3], dim=1
                    )
                    mask_loss = criterion(mask_outs, mask_labels)
                    gender_loss = criterion(gender_outs, gender_labels)
                    age_loss = criterion(age_outs, age_labels)
                    loss = mask_loss + gender_loss + age_loss

                    mask_preds = mask_outs.argmax(1)
                    gender_preds = gender_outs.argmax(1)
                    age_preds = age_outs.argmax(1)
                    preds = mask_preds * 6 + gender_preds * 3 + age_preds
                    f1_test += f1_score(
                        preds.cpu().numpy(), labels.cpu().numpy(), average="macro"
                    )

                    val_loss_items.append(loss.item())  # why append?
                    mask_matches += torch.sum(mask_preds == mask_labels).item()
                    gender_matches += torch.sum(gender_preds == gender_labels).item()
                    age_matches += torch.sum(age_preds == age_labels).item()
                    matches += torch.sum(preds == labels).item()

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs)
                            .detach()
                            .cpu()
                            .permute(0, 2, 3, 1)
                            .numpy()
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
                # print(val_mask_items,val_gender_items, val_age_items)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = matches / len(val_set)
                val_mask_acc = mask_matches / len(val_set)
                val_gender_acc = gender_matches / len(val_set)
                val_age_acc = age_matches / len(val_set)
                f1_test = f1_test / len(val_loader)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                    )
                    torch.save(
                        model.module.state_dict(), f"{save_dir}/best_{index}.pth"
                    )  # 최선의 모델
                    best_val_acc = val_acc
                torch.save(
                    model.module.state_dict(), f"{save_dir}/last_{index}.pth"
                )  # 최고 epoch 모델
                print(
                    f"[Val] acc : {val_acc:4.2%}, Mask acc: {val_mask_acc:4.2%}, Gender acc: {val_gender_acc:4.2%}, Age acc: {val_age_acc:4.2%},F1: {f1_test:4.4} loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )

                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1", f1_test, epoch)

                logger.add_figure("results", figure, epoch)
                print()


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
        "--optimizer", type=str, default="Adam", help="optimizer type (default: SGD)"
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
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
