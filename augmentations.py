from typing import Any

import cv2
import numpy as np
import torch
import albumentations as A
import albumentations.pytorch as APT
import torchvision.transforms.v2 as transforms


class _RandAugment(transforms.RandAugment):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }


class RandAugment(A.ImageOnlyTransform):
    def __init__(self, magnitude: int = 9, n: int = 2, always_apply: bool = False, p: float = 1.0) -> None:
        super().__init__(always_apply, p)
        self.magnitude = magnitude
        self.n = n
        self.augmentation = _RandAugment(num_ops=n, magnitude=magnitude)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        rgb_tensor = torch.from_numpy(img).permute(2, 0, 1)
        augmented: torch.Tensor = self.augmentation(rgb_tensor)
        return augmented.permute(1, 2, 0).numpy()


class BlackoutCrop(A.ImageOnlyTransform):
    def __init__(self, height: int, width: int, scale: tuple[float, float], ratio: tuple[float, float], always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
    
    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        new_img = np.zeros_like(img)
        torch_img = torch.from_numpy(img).permute(2, 0, 1)
        i, j, h, w = transforms.RandomResizedCrop.get_params(torch_img, scale=self.scale, ratio=self.ratio)
        new_img[j:j+h, i:i+w] = img[j:j+h, i:i+w]
        return new_img


def _low_quality_rescale_augmentation(p: float) -> A.OneOf:
    interpolations = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4)
    return A.OneOf([
        A.Compose([
            A.RandomScale(scale_limit=(0.2, 0.5), interpolation=shrink_interpolation, p=1.0),
            A.Resize(112, 112, interpolation=enlarge_interpolation, p=1.0),
        ], p=1.0)
        for shrink_interpolation in interpolations
        for enlarge_interpolation in interpolations
    ], p=p)


def get_transform(kind: str | None = None) -> A.Compose:
    base_transform = A.Compose([
        A.Resize(112, 112, interpolation=cv2.INTER_AREA),
        A.ToFloat(max_value=255.0),
        APT.ToTensorV2(),
    ], additional_targets={"image2": "image"})
    if kind == "digiface":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 9), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.MotionBlur(blur_limit=(3, 9), p=0.1),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0, p=0.2),
            BlackoutCrop(112, 112, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=0.2),
            A.ToGray(p=0.1),
            _low_quality_rescale_augmentation(p=0.3),
            A.ImageCompression(quality_lower=20, quality_upper=80, p=0.2),
            base_transform,
        ], additional_targets={"image2": "image"})
    elif kind == "adaface":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0, p=0.2),
            BlackoutCrop(112, 112, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=0.2),
            A.ToGray(p=0.1),
            _low_quality_rescale_augmentation(p=0.2),
            A.ImageCompression(quality_lower=20, quality_upper=80, p=0.2),
            base_transform,
        ], additional_targets={"image2": "image"})
    elif kind == "rand":
        return A.Compose([
            RandAugment(magnitude=16, n=4, p=1.0),
            base_transform,
        ], additional_targets={"image2": "image"})
    elif kind is None:
        return base_transform
    else:
        raise NotImplementedError(f"Unknown transform kind: {kind}")