import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)
from torchvision import transforms

from sklearn.model_selection import train_test_split
import copy

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class NoResize:
    def __init__(self, mean, std, **args):
        self.transform = Compose(
            [
                Resize([224, 224], Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        tensor = self.transform(image)
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        return tensor_normalized


class AddGaussianNoise(object):
    """
    transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
    직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                # ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                # AddGaussianNoise()
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class MyAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                transforms.RandomHorizontalFlip(1),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class Grayscale:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 57:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 + 2 + 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)  # target? 이걸 기준으로 slicing해야한다
        for profile in profiles:  # 1. images안의 파일을 iterating
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):  # 2. iterating한 폴더 내 이미지 iterating
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[
                    _file_name
                ]  # mask label preprocess로 보면됨 mask1 -> 0

                id, gender, race, age = profile.split(
                    "_"
                )  # 3. 폴더 iterating할때 폴더명 _ 기준 slicing
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
        # 즉 위 코드는 그냥 라벨값만 더해주는 코드이다
        # 아마 이쪽에 폴더별 slicing기능을 넣어주어야할듯
        # 전처리과정 손보고 다시 검증하기는 시간이 걸릴 느낌이니 다 해놓은거를 아래쪽에서 슬라이싱 해보는걸로

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        # three_class_label = torch.Tensor([mask_label, gender_label, age_label])  #Ai stage의 데이터칸 순서와 동일

        image_transform = self.transform(image)
        return image_transform, (mask_label, gender_label, age_label)

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        # n_val = int(len(self) * self.val_ratio)
        # n_train = len(self) - n_val
        train_indices, val_indices = train_test_split(
            range(len(self)), test_size=self.val_ratio
        )
        self_ = copy.deepcopy(self)
        train_set = Subset(self, train_indices)
        val_set = Subset(self_, val_indices)
        # train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌
    사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> Tuple[Subset, Subset]:
        # train_indices, val_indices = train_test_split(range(len(self)), test_size=self.val_ratio)
        self_ = copy.deepcopy(self)
        indices = list(self.indices.items())
        train_set = Subset(self, indices[0][1])
        val_set = Subset(self_, indices[1][1])
        # train_set, val_set = random_split(self, [n_train, n_val])
        # return [Subset(self, indices) for phase, indices in self.indices.items()]
        return [train_set, val_set]


class TestDataset(Dataset):
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    def set_transform(self, transform):
        self.transform = transform


class TTADataset(Dataset):
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, p=0.5),
                transforms.CenterCrop(size=(224, 224), p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2
                ),
                transforms.Resize(224, 224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
                ),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
