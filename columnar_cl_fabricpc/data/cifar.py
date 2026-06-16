"""
CIFAR-10 data loading utilities.

Downloads and loads CIFAR-10 dataset without torchvision dependency.
Uses direct download from the official source and numpy for processing.
"""

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Tuple

import numpy as np

# CIFAR-10 download URL and file info
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-py"


class Batch(NamedTuple):
    """A batch of images and labels."""

    images: np.ndarray  # Shape: (batch_size, 32, 32, 3), dtype: float32
    labels: np.ndarray  # Shape: (batch_size,), dtype: int32


class CIFAR10Dataset(NamedTuple):
    """CIFAR-10 dataset splits."""

    train_images: np.ndarray  # Shape: (N_train, 32, 32, 3)
    train_labels: np.ndarray  # Shape: (N_train,)
    test_images: np.ndarray  # Shape: (10000, 32, 32, 3)
    test_labels: np.ndarray  # Shape: (10000,)


def get_data_dir() -> Path:
    """Get the data directory, creating if needed."""
    # Use XDG_DATA_HOME if set, otherwise ~/.local/share
    xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    data_dir = Path(xdg_data) / "columnar_cl_fabricpc" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_cifar10(data_dir: Optional[Path] = None) -> Path:
    """
    Download CIFAR-10 dataset if not already present.

    Args:
        data_dir: Directory to store the data. Defaults to XDG data directory.

    Returns:
        Path to the extracted CIFAR-10 folder.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    cifar_folder = data_dir / CIFAR10_FOLDER

    if cifar_folder.exists():
        return cifar_folder

    tar_path = data_dir / CIFAR10_FILENAME

    if not tar_path.exists():
        print(f"Downloading CIFAR-10 to {tar_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        print("Download complete.")

    print(f"Extracting CIFAR-10 to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")
    print("Extraction complete.")

    # Remove tar file to save space
    tar_path.unlink()

    return cifar_folder


def _load_batch(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch file."""
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # Data is stored as (10000, 3072) uint8, reshape to (N, 3, 32, 32)
    images = batch[b"data"].reshape(-1, 3, 32, 32)
    # Transpose to (N, 32, 32, 3) for channel-last format
    images = images.transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"], dtype=np.int32)

    return images, labels


def load_cifar10(
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> CIFAR10Dataset:
    """
    Load the CIFAR-10 dataset.

    Args:
        data_dir: Directory containing CIFAR-10 data. If None, downloads to
            default location.
        normalize: If True, normalize pixel values from [0, 255] to normalize_range.
        normalize_range: Target range for normalization. Default is [-1, 1].

    Returns:
        CIFAR10Dataset with train and test splits.
    """
    cifar_folder = download_cifar10(data_dir)

    # Load training batches (5 batches of 10000 images each)
    train_images_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = cifar_folder / f"data_batch_{i}"
        images, labels = _load_batch(batch_path)
        train_images_list.append(images)
        train_labels_list.append(labels)

    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # Load test batch
    test_path = cifar_folder / "test_batch"
    test_images, test_labels = _load_batch(test_path)

    # Convert to float32
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    if normalize:
        low, high = normalize_range
        # Normalize from [0, 255] to [low, high]
        train_images = train_images / 255.0 * (high - low) + low
        test_images = test_images / 255.0 * (high - low) + low

    return CIFAR10Dataset(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


class DataLoader:
    """
    Batch iterator for CIFAR-10 data.

    Yields batches of (images, labels) for training or evaluation.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the data loader.

        Args:
            images: Image array of shape (N, 32, 32, 3).
            labels: Label array of shape (N,).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data each epoch.
            drop_last: If True, drop the last incomplete batch.
            rng: Random number generator for shuffling.
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = rng if rng is not None else np.random.default_rng()

        self.n_samples = len(images)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.n_batches

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        indices = np.arange(self.n_samples)

        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size

            if end > self.n_samples:
                if self.drop_last:
                    break
                end = self.n_samples

            batch_indices = indices[start:end]
            yield Batch(
                images=self.images[batch_indices],
                labels=self.labels[batch_indices],
            )


def create_data_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for CIFAR-10.

    Args:
        batch_size: Number of samples per batch.
        val_split: Fraction of training data to use for validation.
        data_dir: Directory containing CIFAR-10 data.
        normalize: If True, normalize pixel values.
        normalize_range: Target range for normalization.
        seed: Random seed for train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset = load_cifar10(
        data_dir=data_dir,
        normalize=normalize,
        normalize_range=normalize_range,
    )

    # Split training data into train and validation
    rng = np.random.default_rng(seed)
    n_train = len(dataset.train_images)
    n_val = int(n_train * val_split)
    n_train_actual = n_train - n_val

    indices = np.arange(n_train)
    rng.shuffle(indices)

    train_indices = indices[:n_train_actual]
    val_indices = indices[n_train_actual:]

    train_images = dataset.train_images[train_indices]
    train_labels = dataset.train_labels[train_indices]
    val_images = dataset.train_images[val_indices]
    val_labels = dataset.train_labels[val_indices]

    train_loader = DataLoader(
        images=train_images,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        rng=np.random.default_rng(seed),
    )

    val_loader = DataLoader(
        images=val_images,
        labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        images=dataset.test_images,
        labels=dataset.test_labels,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
