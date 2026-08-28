"""
MNIST, downloaded and parsed without torchvision.

torchvision is deliberately not used: importing it (like importing
torch.onnx, which torch.optim reaches through torch._dynamo) pulls in
whatever `transformers` is installed, and a mismatched one takes the whole
process down. The IDX format is simple enough that reading it directly costs
less than depending on that, and it keeps the dataset a plain pair of
in-memory tensors -- which is also the fastest thing to train on, since the
whole of MNIST is only ~180MB as float32 and never has to touch the disk
again after the first epoch.

Files are fetched from the S3 mirror (the canonical yann.lecun.com host
frequently refuses automated requests) into <data_dir>/raw/ and left there;
subsequent runs reuse them. Images are scaled to [0, 1] and standardised
with the usual MNIST statistics.
"""

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# The standard MNIST channel statistics, applied after scaling to [0, 1].
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def download(data_dir: Path) -> Path:
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = raw_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        url = MIRROR + filename
        print(f"Downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)
    return raw_dir


def _read_idx(path: Path) -> np.ndarray:
    """Parse an IDX file: big-endian magic (zero, zero, dtype, n_dims), then
    one big-endian int32 per dimension, then the raw data. Only the uint8
    dtype (0x08) occurs in MNIST."""
    with gzip.open(path, "rb") as f:
        magic = f.read(4)
        zero1, zero2, dtype_code, n_dims = struct.unpack(">BBBB", magic)
        if (zero1, zero2, dtype_code) != (0, 0, 0x08):
            raise ValueError(f"{path.name}: unexpected IDX header {magic!r}; expected a uint8 tensor")
        shape = struct.unpack(f">{n_dims}I", f.read(4 * n_dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path.name}: expected {expected} bytes of data, got {data.size}")
    return data.reshape(shape)


def build_mnist_datasets(data_dir: Path) -> Tuple[TensorDataset, TensorDataset]:
    """(train, test) as in-memory TensorDatasets of normalised (1, 28, 28)
    images and int64 labels. Downloads on first use."""
    raw_dir = download(data_dir)

    def to_dataset(images_file: str, labels_file: str) -> TensorDataset:
        images = _read_idx(raw_dir / images_file).astype(np.float32) / 255.0
        images = (images - MNIST_MEAN) / MNIST_STD
        labels = _read_idx(raw_dir / labels_file).astype(np.int64)
        if images.shape[0] != labels.shape[0]:
            raise ValueError(f"{images_file} has {images.shape[0]} images but {labels_file} has {labels.shape[0]} labels")
        return TensorDataset(
            torch.from_numpy(images).unsqueeze(1),
            torch.from_numpy(labels),
        )

    train = to_dataset(FILES["train_images"], FILES["train_labels"])
    test = to_dataset(FILES["test_images"], FILES["test_labels"])
    return train, test
