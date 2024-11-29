import csv
import os

import PIL.Image as PImage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def normalize_255_into_pm1(x):
    return x.div_(127.5).sub(-1)


class COCODataset(Dataset):
    def __init__(self, root_dir, subset_name="subset", transform=None, max_cnt=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            "jpg",
            "jpeg",
            "png",
            "ppm",
            "bmp",
            "pgm",
            "tif",
            "tiff",
            "webp",
        )
        sample_dir = os.path.join(root_dir, subset_name)

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.split('.')[-1] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        # restrict num samples
        self.samples = self.samples if max_cnt is None else self.samples[:max_cnt]  

        # Collect captions
        self.captions = {}
        with open(
            os.path.join(root_dir, f"{subset_name}.csv"), newline="\n"
        ) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return sample, self.captions[os.path.basename(sample_path)]


def coco_collate_fn(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def build_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
):
    # build augmentations
    # first resize to mid_reso, then crop to final_reso
    mid_reso = round(mid_reso * final_reso)
    train_aug = [
        transforms.Resize(
            mid_reso,
            interpolation=InterpolationMode.LANCZOS,
        ),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug = transforms.Compose(train_aug)

    # build dataset
    train_set = COCODataset(
        data_path,
        subset_name="train2014",
        transform=train_aug,
        max_cnt=None,
    )
    print(f"[Dataset] {len(train_set)=}")
    print_aug(train_aug, "[train]")

    return train_set


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def print_aug(transform, label):
    print(f"Transform {label} = ")
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print("---------------------------\n")
