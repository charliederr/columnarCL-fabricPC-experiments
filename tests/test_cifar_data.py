"""
Tests for CIFAR-10 data loading utilities.

These tests verify that CIFAR-10 loading, normalization, and batching work
correctly. The first run will download CIFAR-10 (~170MB).
"""

import numpy as np
import pytest

from columnar_cl_fabricpc.data import (
    CIFAR10_CLASSES,
    Batch,
    DataLoader,
    create_data_loaders,
    load_cifar10,
)


class TestLoadCIFAR10:
    """Test CIFAR-10 dataset loading."""

    @classmethod
    @pytest.fixture(scope="class")
    def dataset(cls):
        """Load CIFAR-10 once for all tests in this class."""
        return load_cifar10(normalize=True, normalize_range=(-1.0, 1.0))

    def test_train_images_shape(self, dataset):
        """Training images should be (50000, 32, 32, 3)."""
        assert dataset.train_images.shape == (50000, 32, 32, 3)

    def test_train_labels_shape(self, dataset):
        """Training labels should be (50000,)."""
        assert dataset.train_labels.shape == (50000,)

    def test_test_images_shape(self, dataset):
        """Test images should be (10000, 32, 32, 3)."""
        assert dataset.test_images.shape == (10000, 32, 32, 3)

    def test_test_labels_shape(self, dataset):
        """Test labels should be (10000,)."""
        assert dataset.test_labels.shape == (10000,)

    def test_images_dtype(self, dataset):
        """Images should be float32."""
        assert dataset.train_images.dtype == np.float32
        assert dataset.test_images.dtype == np.float32

    def test_labels_dtype(self, dataset):
        """Labels should be int32."""
        assert dataset.train_labels.dtype == np.int32
        assert dataset.test_labels.dtype == np.int32

    def test_normalized_range(self, dataset):
        """Normalized images should be in [-1, 1]."""
        assert dataset.train_images.min() >= -1.0
        assert dataset.train_images.max() <= 1.0
        assert dataset.test_images.min() >= -1.0
        assert dataset.test_images.max() <= 1.0

    def test_label_range(self, dataset):
        """Labels should be in [0, 9]."""
        assert dataset.train_labels.min() >= 0
        assert dataset.train_labels.max() <= 9
        assert dataset.test_labels.min() >= 0
        assert dataset.test_labels.max() <= 9

    def test_all_classes_present(self, dataset):
        """All 10 classes should be present in training data."""
        unique_labels = set(dataset.train_labels)
        assert unique_labels == set(range(10))


class TestNormalization:
    """Test different normalization options."""

    def test_no_normalization(self):
        """Without normalization, images should be in [0, 255]."""
        dataset = load_cifar10(normalize=False)
        assert dataset.train_images.min() >= 0.0
        assert dataset.train_images.max() <= 255.0

    def test_zero_one_normalization(self):
        """Test normalization to [0, 1] range."""
        dataset = load_cifar10(normalize=True, normalize_range=(0.0, 1.0))
        assert dataset.train_images.min() >= 0.0
        assert dataset.train_images.max() <= 1.0


class TestDataLoader:
    """Test the DataLoader class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        images = np.random.randn(100, 32, 32, 3).astype(np.float32)
        labels = np.random.randint(0, 10, size=100, dtype=np.int32)
        return images, labels

    def test_batch_count(self, sample_data):
        """Test correct number of batches."""
        images, labels = sample_data
        loader = DataLoader(images, labels, batch_size=32, drop_last=True)
        assert len(loader) == 3  # 100 // 32 = 3

    def test_batch_count_no_drop(self, sample_data):
        """Test batch count without dropping last."""
        images, labels = sample_data
        loader = DataLoader(images, labels, batch_size=32, drop_last=False)
        assert len(loader) == 4  # ceil(100 / 32) = 4

    def test_batch_shapes(self, sample_data):
        """Test that batches have correct shapes."""
        images, labels = sample_data
        loader = DataLoader(images, labels, batch_size=32, drop_last=True)

        for batch in loader:
            assert isinstance(batch, Batch)
            assert batch.images.shape == (32, 32, 32, 3)
            assert batch.labels.shape == (32,)

    def test_last_batch_smaller(self, sample_data):
        """Test that last batch can be smaller when not dropping."""
        images, labels = sample_data
        loader = DataLoader(images, labels, batch_size=32, drop_last=False)

        batches = list(loader)
        assert batches[-1].images.shape[0] == 100 - 3 * 32  # 4 remaining

    def test_shuffle_changes_order(self, sample_data):
        """Test that shuffle produces different batch orders."""
        images, labels = sample_data

        loader1 = DataLoader(
            images, labels, batch_size=32, shuffle=True, rng=np.random.default_rng(1)
        )
        loader2 = DataLoader(
            images, labels, batch_size=32, shuffle=True, rng=np.random.default_rng(2)
        )

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Different seeds should produce different first batches
        assert not np.allclose(batch1.images, batch2.images)

    def test_no_shuffle_preserves_order(self, sample_data):
        """Test that no shuffle preserves order."""
        images, labels = sample_data

        loader = DataLoader(images, labels, batch_size=32, shuffle=False)

        batch = next(iter(loader))
        assert np.allclose(batch.images, images[:32])
        assert np.array_equal(batch.labels, labels[:32])


class TestCreateDataLoaders:
    """Test the create_data_loaders function."""

    def test_creates_three_loaders(self):
        """Test that three loaders are created."""
        train, val, test = create_data_loaders(batch_size=128)

        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_val_split(self):
        """Test that validation split is correct size."""
        train, val, test = create_data_loaders(batch_size=128, val_split=0.1)

        # With 50000 training images and 0.1 val split:
        # train: 45000, val: 5000
        train_samples = sum(len(b.labels) for b in train)
        val_samples = sum(len(b.labels) for b in val)

        # Training drops last batch, so check range
        assert 44800 <= train_samples <= 45000
        assert val_samples == 5000

    def test_test_size(self):
        """Test that test set has 10000 images."""
        train, val, test = create_data_loaders(batch_size=128)

        test_samples = sum(len(b.labels) for b in test)
        assert test_samples == 10000


class TestCIFAR10Classes:
    """Test CIFAR-10 class names."""

    def test_ten_classes(self):
        """There should be 10 class names."""
        assert len(CIFAR10_CLASSES) == 10

    def test_known_classes(self):
        """Check for expected class names."""
        assert "airplane" in CIFAR10_CLASSES
        assert "cat" in CIFAR10_CLASSES
        assert "truck" in CIFAR10_CLASSES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
