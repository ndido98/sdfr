from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
import os
import random
import joblib

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
import albumentations.pytorch as APT

from augmentations import get_transform
from containers import SupervisedArray, SupervisedCoupleArray


def _load_image(file: Path) -> np.ndarray:
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not load image {file}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _find_images(root_dir: Path) -> list[Path]:
    return [root_dir / f for f in os.scandir(root_dir) if f.is_file() and f.name.endswith((".png", ".jpg"))]


class FaceClassificationDataset(ABC, Dataset):
    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass


class SingleFaceClassificationDataset(FaceClassificationDataset):
    def __init__(
        self,
        root_dir: Path,
        indices_file: Path | None = None,
        from_class: int | None = None,
        to_class: int | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        if indices_file is not None:
            # One line per class, each line contains the names of the images to use
            all_classes = indices_file.read_text().strip().splitlines()
            if from_class is None and to_class is None:
                classes = all_classes
            elif from_class is None:
                classes = all_classes[:to_class]
            elif to_class is None:
                classes = all_classes[from_class:]
            else:
                classes = all_classes[from_class:to_class]
            self._n_classes = len(classes)
            files = []
            for i, klass in enumerate(classes):
                images = klass.split("\t")
                for image in images:
                    files.append((str(root_dir / image), i))
        else:
            all_classes = sorted([c.name for c in os.scandir(root_dir) if c.is_dir()])
            if from_class is None and to_class is None:
                classes = all_classes
            elif from_class is None:
                classes = all_classes[:to_class]
            elif to_class is None:
                classes = all_classes[from_class:]
            else:
                classes = all_classes[from_class:to_class]
            self._n_classes = len(classes)
            files = []
            classes_images = joblib.Parallel(n_jobs=-1)(joblib.delayed(_find_images)(root_dir / str(klass)) for klass in classes)
            files = [(str(f), i) for i, class_images in enumerate(classes_images) for f in class_images]
        self.files = SupervisedArray(files)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        file, klass = self.files[idx]
        img = _load_image(file)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, klass


class JointFaceClassificationDataset(FaceClassificationDataset):
    def __init__(self, datasets: list[FaceClassificationDataset]) -> None:
        self.datasets = datasets
        self.datasets_lengths = [len(d) for d in self.datasets]

    @property
    def n_classes(self) -> int:
        return sum(d.n_classes for d in self.datasets)

    def __len__(self) -> int:
        return sum(self.datasets_lengths)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        dataset_idx = np.searchsorted(np.cumsum(self.datasets_lengths), idx, side="right")
        sample_idx = idx - sum(self.datasets_lengths[:dataset_idx])
        file, original_class = self.datasets[dataset_idx][sample_idx]
        shifted_class = sum(d.n_classes for d in self.datasets[:dataset_idx]) + original_class
        return file, shifted_class


class FaceCouplesDataset(FaceClassificationDataset):
    def __init__(
        self,
        root_dir: Path,
        indices_file: Path | None = None,
        from_class: int | None = None,
        to_class: int | None = None,
        max_matches_per_image: int = 3,
        max_nonmatches_per_image: int = 3,
        max_images_per_class: int | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        if indices_file is not None:
            all_classes = indices_file.read_text().strip().splitlines()
            if from_class is None and to_class is None:
                classes = all_classes
            elif from_class is None:
                classes = all_classes[:to_class]
            elif to_class is None:
                classes = all_classes[from_class:]
            else:
                classes = all_classes[from_class:to_class]
            self._n_classes = len(classes)
            files: dict[str, list[Path]] = {}
            for i, klass in enumerate(classes):
                images = klass.split("\t")
                files[klass] = [(str(root_dir / image), i) for image in images]
        else:
            all_classes = sorted([c.name for c in os.scandir(root_dir) if c.is_dir()])
            if from_class is None and to_class is None:
                classes = all_classes
            elif from_class is None:
                classes = all_classes[:to_class]
            elif to_class is None:
                classes = all_classes[from_class:]
            else:
                classes = all_classes[from_class:to_class]
            self._n_classes = len(classes)
            files: dict[str, list[Path]] = {}
            for klass in classes:
                class_dir = root_dir / str(klass)
                files[klass] = [
                    (class_dir / f, klass)
                    for f in os.scandir(class_dir)
                    if f.is_file() and f.name.endswith((".png", ".jpg"))
                ]
        matching_couples, non_matching_couples = set(), set()
        for klass in classes:
            # Add the matching samples
            class_files = files[klass]
            if max_images_per_class is not None and len(class_files) > max_images_per_class:
                matching_class_files = random.sample(class_files, max_images_per_class)
                non_matching_class_files = random.sample(class_files, max_images_per_class)
            else:
                matching_class_files = class_files
                non_matching_class_files = class_files
            for file1, _ in matching_class_files:
                matches = 0
                while matches < max_matches_per_image:
                    file2 = random.choice(matching_class_files)[0]
                    if file1 == file2:
                        continue
                    couple = ((str(file1), str(file2)), 1)
                    if couple not in matching_couples:
                        matching_couples.add(couple)
                        matches += 1
            # Add the non-matching samples
            other_classes = [c for c in classes if c != klass]
            for file1, _ in non_matching_class_files:
                non_matches = 0
                while non_matches < max_nonmatches_per_image:
                    chosen_class = random.choice(other_classes)
                    file2 = random.choice(files[chosen_class])[0]
                    couple = ((str(file1), str(file2)), 0)
                    if couple not in non_matching_couples:
                        non_matching_couples.add(couple)
                        non_matches += 1
        couples = list(matching_couples) + list(non_matching_couples)
        random.shuffle(couples)
        self.couples = SupervisedCoupleArray(couples)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def __len__(self) -> int:
        return len(self.couples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (file1, file2), is_same = self.couples[idx]
        img1 = _load_image(file1)
        img2 = _load_image(file2)
        if self.transform:
            result = self.transform(image=img1, image2=img2)
            img1, img2 = result["image"], result["image2"]
        return img1, img2, is_same


class JointFaceCouplesDataset(FaceClassificationDataset):
    def __init__(self, datasets: list[FaceCouplesDataset]) -> None:
        self.datasets = datasets
        self.datasets_lengths = [len(d) for d in self.datasets]

    @property
    def n_classes(self) -> int:
        return sum(d.n_classes for d in self.datasets)

    def __len__(self) -> int:
        return sum(self.datasets_lengths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        dataset_idx = np.searchsorted(np.cumsum(self.datasets_lengths), idx, side="right")
        sample_idx = idx - sum(self.datasets_lengths[:dataset_idx])
        file1, file2, is_same = self.datasets[dataset_idx][sample_idx]
        return file1, file2, is_same


class CouplesFileDataset(Dataset):
    def __init__(self, root_dir: Path, couples_file: Path, transform: Callable | None = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        file_lines = couples_file.read_text().strip().splitlines()
        file_couples = [line.split("\t") for line in file_lines]
        if len(file_couples) == 0:
            raise ValueError(f"File {couples_file} is empty")
        if len(file_couples[0]) == 2:
            self.couples = [(Path(c1), Path(c2)) for c1, c2 in file_couples]
        else:
            self.couples = [(Path(c1), Path(c2), int(s)) for c1, c2, s in file_couples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.couples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        couple = self.couples[idx]
        if len(couple) == 3:
            img1, img2, klass = couple
        else:
            img1, img2 = couple
        img1 = _load_image(self.root_dir / img1)
        img2 = _load_image(self.root_dir / img2)
        if self.transform:
            result = self.transform(image=img1, image2=img2)
            img1, img2 = result["image"], result["image2"]
        if len(couple) == 3:
            return img1, img2, klass
        else:
            return img1, img2


@dataclass
class TrainDataset:
    root: str
    indices_file: str | None = None
    train_from_class: int | None = None
    train_to_class: int | None = None
    val_from_class: int | None = None
    val_to_class: int | None = None
    val_max_images_per_class: int | None = None
    val_max_matches_per_image: int = 4
    val_max_nonmatches_per_image: int = 4
    augment: Literal["adaface", "rand", "digiface"] | None = None,


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_datasets: list[TrainDataset],
        test_dataset_root: str,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.train_datasets = train_datasets
        self.test_dataset_root = Path(test_dataset_root)
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        train_datasets = []
        val_datasets = []
        base_transform = get_transform(None)
        for ds in self.train_datasets:
            if stage == "fit" and ds.train_from_class is not None or ds.train_to_class is not None:
                transform = get_transform(ds.augment)
                train_ds = SingleFaceClassificationDataset(
                    Path(ds.root),
                    indices_file=Path(ds.indices_file) if ds.indices_file is not None else None,
                    from_class=ds.train_from_class,
                    to_class=ds.train_to_class,
                    transform=transform,
                )
                print(f"Added training set {ds.root} with {len(train_ds)} samples across {train_ds.n_classes} classes")
                train_datasets.append(train_ds)
            if stage in ("fit", "validate") and ds.val_from_class is not None or ds.val_to_class is not None:
                val_ds = FaceCouplesDataset(
                    Path(ds.root),
                    indices_file=Path(ds.indices_file) if ds.indices_file is not None else None,
                    from_class=ds.val_from_class,
                    to_class=ds.val_to_class,
                    max_images_per_class=ds.val_max_images_per_class,
                    max_matches_per_image=ds.val_max_matches_per_image,
                    max_nonmatches_per_image=ds.val_max_nonmatches_per_image,
                    transform=base_transform,
                )
                print(f"Added validation set {ds.root} with {len(val_ds)} samples across {val_ds.n_classes} classes")
                val_datasets.append(val_ds)
        if stage == "fit":
            assert len(train_datasets) > 0
            self.train_dataset = JointFaceClassificationDataset(train_datasets)
            assert len(self.train_dataset) > 0
            print(f"Training set has {len(self.train_dataset)} samples across {self.train_dataset.n_classes} classes")
        if stage in ("fit", "validate"):
            assert len(val_datasets) > 0
            self.val_dataset = JointFaceCouplesDataset(val_datasets)
            assert len(self.val_dataset) > 0
            print(f"Validation set has {len(self.val_dataset)} samples across {self.val_dataset.n_classes} classes")
        if stage == "test":
            self.test_dataset = CouplesFileDataset(
                self.test_dataset_root,
                self.test_dataset_root / "test_pairs.txt",
                transform=base_transform,
            )
            print(f"Test set has {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
