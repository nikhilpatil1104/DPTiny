import gzip
import os
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

from dptiny.backend import xp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")
_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _parse_idx(raw: bytes) -> np.ndarray:
    if len(raw) < 4:
        raise ValueError("File is too short to be an IDX file")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError("Invalid IDX magic number")
    if raw[2] != 0x08:
        raise ValueError("IDX data type is not uint8")
    ndim = raw[3]
    if ndim == 0:
        raise ValueError("IDX file has no dimensions")
    header_size = 4 + 4 * ndim
    if len(raw) < header_size:
        raise ValueError("IDX header is truncated")
    shape = tuple(
        int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)
    )
    expected = int(np.prod(shape, dtype=np.int64))
    if len(raw) - header_size != expected:
        raise ValueError("IDX data size does not match its header")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


def _read_idx(path: str) -> np.ndarray:
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"{path} is not a valid gzip file") from exc
    return _parse_idx(raw)


def _download(filename: str, cache_dir: str) -> str:
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            _BASE_URL + filename
        ) as response:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    cache_dir = os.path.join(cache_dir, "fashion-mnist")
    os.makedirs(cache_dir, exist_ok=True)

    arrays = {
        key: _read_idx(_download(filename, cache_dir))
        for key, filename in _FILES.items()
    }

    X_train, X_test = arrays["train_images"], arrays["test_images"]
    y_train, y_test = arrays["train_labels"], arrays["test_labels"]
    if (
        X_train.shape != (60000, 28, 28)
        or X_test.shape != (10000, 28, 28)
        or y_train.shape != (60000,)
        or y_test.shape != (10000,)
    ):
        raise ValueError("Unexpected Fashion-MNIST array shapes")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32)
