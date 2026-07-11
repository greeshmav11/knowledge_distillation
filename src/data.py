"""
Data loading for CIFAR-100 and CIFAR-100-LT (long-tailed).

Key pieces:
  - get_cifar100_loaders(...): standard balanced CIFAR-100 train/test loaders
  - get_cifar100_lt_loaders(...): long-tailed (imbalanced) training set,
    balanced test set, using the standard exponential imbalance profile
    from Cao et al. 2019 / Cui et al. 2019.
  - get_fine_to_coarse_mapping(root): reads the *raw* CIFAR-100 pickle files
    to build the fine-label -> superclass (20 coarse classes) mapping
    directly from the dataset itself, rather than relying on a hardcoded
    (and easy-to-get-wrong) lookup table.

Usage:
    from data import get_cifar100_loaders, get_cifar100_lt_loaders, get_fine_to_coarse_mapping

    train_loader, test_loader, meta = get_cifar100_loaders(root="./data", batch_size=128)
    fine_to_coarse = get_fine_to_coarse_mapping(root="./data")
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import torchvision
import torchvision.transforms as T


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_cifar100_loaders(root="./data", batch_size=128, num_workers=4):
    """Standard, class-balanced CIFAR-100."""
    train_set = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=_train_transform()
    )
    test_set = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=_test_transform()
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    meta = {"num_classes": 100, "class_names": train_set.classes}
    return train_loader, test_loader, meta


class _IndexSubsetDataset(torch.utils.data.Dataset):
    """Wraps a base dataset and restricts it to a fixed list of indices."""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]


def _build_lt_indices(labels, num_classes, imb_factor=0.01, seed=0):
    """
    Build indices for a long-tailed subset following the standard
    exponential-decay imbalance profile used in CIFAR-LT papers.

    imb_factor = (# examples in smallest class) / (# examples in largest class)
    A common choice is imb_factor=0.01 (i.e. imbalance ratio 100) or 0.1 (ratio 10).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    n_per_class_original = len(labels) // num_classes  # 500 for CIFAR-100 train

    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = n_per_class_original * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))

    selected_indices = []
    class_counts = {}
    for cls_idx, n_keep in enumerate(img_num_per_cls):
        cls_indices = np.where(labels == cls_idx)[0]
        rng.shuffle(cls_indices)
        keep = cls_indices[:n_keep]
        selected_indices.extend(keep.tolist())
        class_counts[cls_idx] = len(keep)

    rng.shuffle(selected_indices)
    return selected_indices, class_counts


def get_cifar100_lt_loaders(root="./data", batch_size=128, num_workers=4,
                             imb_factor=0.01, seed=0):
    """
    Long-tailed CIFAR-100 training set + standard balanced CIFAR-100 test set.
    Returns train_loader, test_loader, meta (meta includes 'class_counts',
    a dict {class_idx: num_train_examples} -- this is your headline
    "class frequency" variable for the failure-correlate analysis).
    """
    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=_train_transform()
    )
    test_set = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=_test_transform()
    )

    labels = base_train.targets
    lt_indices, class_counts = _build_lt_indices(
        labels, num_classes=100, imb_factor=imb_factor, seed=seed
    )
    lt_train = _IndexSubsetDataset(base_train, lt_indices)

    train_loader = DataLoader(
        lt_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    meta = {
        "num_classes": 100,
        "class_names": base_train.classes,
        "class_counts": class_counts,
    }
    return train_loader, test_loader, meta


def get_fine_to_coarse_mapping(root="./data"):
    """
    Reads the raw CIFAR-100 pickle (downloaded by torchvision) to build the
    fine_label -> coarse_label (20 superclasses) mapping directly from data,
    which avoids hardcoding an error-prone lookup table.

    Returns:
        fine_to_coarse: np.ndarray of shape (100,), fine_to_coarse[i] = coarse id
        coarse_names: list[str] of length 20
        fine_names: list[str] of length 100
    """
    data_dir = os.path.join(root, "cifar-100-python")
    if not os.path.isdir(data_dir):
        # trigger download via torchvision if not already present
        torchvision.datasets.CIFAR100(root=root, train=True, download=True)

    with open(os.path.join(data_dir, "train"), "rb") as f:
        train_dict = pickle.load(f, encoding="latin1")
    with open(os.path.join(data_dir, "meta"), "rb") as f:
        meta_dict = pickle.load(f, encoding="latin1")

    fine_labels = np.array(train_dict["fine_labels"])
    coarse_labels = np.array(train_dict["coarse_labels"])

    fine_to_coarse = np.zeros(100, dtype=np.int64)
    for fine_id in range(100):
        matches = coarse_labels[fine_labels == fine_id]
        assert len(np.unique(matches)) == 1, "inconsistent fine->coarse mapping found"
        fine_to_coarse[fine_id] = matches[0]

    fine_names = meta_dict["fine_label_names"]
    coarse_names = meta_dict["coarse_label_names"]
    return fine_to_coarse, coarse_names, fine_names


if __name__ == "__main__":
    # quick smoke test
    train_loader, test_loader, meta = get_cifar100_loaders()
    print("standard CIFAR-100:", len(train_loader.dataset), len(test_loader.dataset))

    lt_train_loader, lt_test_loader, lt_meta = get_cifar100_lt_loaders(imb_factor=0.01)
    print("CIFAR-100-LT train size:", len(lt_train_loader.dataset))
    print("min/max class count:", min(lt_meta["class_counts"].values()),
          max(lt_meta["class_counts"].values()))

    fine_to_coarse, coarse_names, fine_names = get_fine_to_coarse_mapping()
    print("example mapping:", fine_names[0], "->", coarse_names[fine_to_coarse[0]])
