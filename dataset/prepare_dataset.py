from typing import Dict, Optional

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from dataset.LITSDataset import LITSDataset
from dataset.transforms.ClampValues import ClampValues
from dataset.transforms.ReshapeTensor import ReshapeTensor


def compose_transforms(config: Dict) -> Compose:  # 数据预处理
    """
    Composes the necessary transforms into lists based on user configuration

    :param config:  dictionary containing configuration instructions
    :return:        dictionary containing lists of transforms and paired transforms
    """
    all_transforms = [
        transforms.ToTensor(),
        ReshapeTensor(),
    ]

    dataset_configs = config["dataset"]

    if dataset_configs["clamp_hu"]:
        min_hu = dataset_configs["clamp_hu_range"]["min"]
        max_hu = dataset_configs["clamp_hu_range"]["max"]
        all_transforms.append(ClampValues((min_hu, max_hu)))

    return transforms.Compose(all_transforms)


def prepare_dataset(config: Dict, train: int) -> LITSDataset:
    """
    Builds the dataset based on user configuration

    :param config:  dictionary containing configuration instructions
    :param train:   int to tell whether to pull training or testing images
    :return:        a created LITSDataset class
    """
    if train == 0:
        img_dirs = config["pathing"]["train_img_dirs"]
        mask_dirs = config["pathing"]["train_mask_dirs"]
    elif train == 1:
        img_dirs = config["pathing"]["test_img_dirs"]
        mask_dirs = config["pathing"]["test_mask_dirs"]
    else:
        img_dirs = config["pathing"]["val_img_dirs"]
        mask_dirs = config["pathing"]["val_mask_dirs"]

    all_transforms = compose_transforms(config)

    dataset = LITSDataset(
        img_dirs, mask_dirs,
        transform=all_transforms,
        config=config)

    return dataset


def prepare_dataloader(config: Dict, train: Optional[int] = 0) -> (DataLoader, int):
    """
    Builds the dataloader class to pass into PyTorch

    :param config:  dictionary containing configuration instructions
    :param train:   int to tell whether to use train or test images
    :return:        DataLoader class with dataset loaded
    """
    dataset = prepare_dataset(config, train)
    if train == 0:
        batch_size = config["dataset"]["batch_size"]
    else:
        batch_size = 1
    shuffle = config["dataset"]["shuffle"]

    num = dataset.img_num
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader, num
