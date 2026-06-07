import torch
import numpy as np
import random 
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

try:
    from torchvision import datasets, transforms
    _TORCHVISION_AVAILABLE = True
except Exception:
    datasets = None
    transforms = None
    _TORCHVISION_AVAILABLE = False


class _TensorTransformPipeline:
    def __init__(self, image_size: int, mean, std, train: bool):
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = train

    @staticmethod
    def _pad(image: torch.Tensor, padding: int = 4) -> torch.Tensor:
        return F.pad(image, (padding, padding, padding, padding), mode="reflect")

    @staticmethod
    def _random_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        _, h, w = image.shape
        if h == size and w == size:
            return image
        if h < size or w < size:
            raise ValueError(f"Cannot crop {size}x{size} from image of shape {(h, w)}")
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return image[:, top:top + size, left:left + size]

    @staticmethod
    def _horizontal_flip(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(image, dims=(2,))
        return image

    @staticmethod
    def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
        if image.shape[-1] == size and image.shape[-2] == size:
            return image
        return F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)

    def __call__(self, image) -> torch.Tensor:
        # torchvision CIFAR datasets yield PIL Images; the custom loaders yield tensors.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        image = image.float() / 255.0
        if self.train:
            if self.image_size == 32:
                image = self._pad(image, padding=4)
                image = self._random_crop(image, 32)
                image = self._horizontal_flip(image)
            else:
                image = self._resize(image, self.image_size)
        else:
            image = self._resize(image, self.image_size)
        image = (image - self.mean) / self.std
        return image


class CIFAR10Dataset(torch.utils.data.Dataset):
    """Minimal CIFAR-10 loader that does not depend on torchvision."""

    base_folder = "cifar-10-batches-py"
    train_files = [f"data_batch_{i}" for i in range(1, 6)]
    test_files = ["test_batch"]
    label_key = "labels"

    def __init__(self, root: str, train: bool, transform=None, download: bool = False):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.data, self.targets = self._load(download=download)

    def _load(self, download: bool = False):
        folder = self.root / self.base_folder
        files = self.train_files if self.train else self.test_files
        if not folder.exists():
            raise FileNotFoundError(
                f"Could not find {folder}. Set DOWNLOAD=1 or place the extracted CIFAR-10 files there."
            )
        data = []
        targets = []
        for filename in files:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 file: {path}")
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            batch = entry.get("data")
            if batch is None:
                raise RuntimeError(f"Invalid CIFAR-10 batch format in {path}")
            data.append(batch.reshape(-1, 3, 32, 32))
            targets.extend(entry[self.label_key])
        data = np.concatenate(data, axis=0)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class CIFAR100Dataset(torch.utils.data.Dataset):
    """Minimal CIFAR-100 loader that does not depend on torchvision."""

    base_folder = "cifar-100-python"
    train_files = ["train"]
    test_files = ["test"]
    label_key = "fine_labels"

    def __init__(self, root: str, train: bool, transform=None, download: bool = False):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.data, self.targets = self._load(download=download)

    def _load(self, download: bool = False):
        folder = self.root / self.base_folder
        files = self.train_files if self.train else self.test_files
        if not folder.exists():
            raise FileNotFoundError(
                f"Could not find {folder}. Set DOWNLOAD=1 or place the extracted CIFAR-100 files there."
            )
        data = []
        targets = []
        for filename in files:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-100 file: {path}")
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            batch = entry.get("data")
            if batch is None:
                raise RuntimeError(f"Invalid CIFAR-100 batch format in {path}")
            data.append(batch.reshape(-1, 3, 32, 32))
            targets.extend(entry[self.label_key])
        data = np.concatenate(data, axis=0)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _subset(dataset, n: int, seed: int):
    if n <= 0 or n >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, idx)

def build_datasets(cfg):
    mean_std = {
        "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }
    mean, std = mean_std[cfg.dataset.upper()]

    train_tfms = _TensorTransformPipeline(cfg.image_size, mean, std, train=True)
    eval_tfms = _TensorTransformPipeline(cfg.image_size, mean, std, train=False)

    if _TORCHVISION_AVAILABLE:
        ds_cls = datasets.CIFAR10 if cfg.dataset.upper() == "CIFAR10" else datasets.CIFAR100
        train_set = ds_cls(cfg.data_dir, train=True, transform=train_tfms, download=cfg.download)
        sens_set = ds_cls(cfg.data_dir, train=True, transform=eval_tfms, download=cfg.download)
        test_set = ds_cls(cfg.data_dir, train=False, transform=eval_tfms, download=cfg.download)
    else:
        ds_cls = CIFAR10Dataset if cfg.dataset.upper() == "CIFAR10" else CIFAR100Dataset
        train_set = ds_cls(cfg.data_dir, train=True, transform=train_tfms, download=cfg.download)
        sens_set = ds_cls(cfg.data_dir, train=True, transform=eval_tfms, download=cfg.download)
        test_set = ds_cls(cfg.data_dir, train=False, transform=eval_tfms, download=cfg.download)

    train_set = _subset(train_set, cfg.train_subset, cfg.seed)
    sens_set = _subset(sens_set, cfg.sensitivity_samples, cfg.seed + 1)
    test_set = _subset(test_set, cfg.test_subset, cfg.seed + 2)
    return train_set, sens_set, test_set