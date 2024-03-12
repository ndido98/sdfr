# coding=utf-8
"""
  @file: datasets.py
  @data: 17 November 2023
  @author: Christophe Ecabert
  @email: christophe.ecabert@idiap.ch
"""
from typing import (Union, Tuple, Iterable, List, Optional, Callable, Any,
                    Sequence, overload, cast)
import pickle
import collections
from io import BytesIO
import numpy as np
import PIL.Image as PImg


DATA_POINT = Tuple[np.ndarray, np.ndarray]
TRANSFORM_FN = Callable[[np.ndarray], np.ndarray]


def _read_bin(filename: str):
    """ Load pickle file """
    with open(filename, 'rb') as f_obj:
        return pickle.load(f_obj, encoding='bytes')


def to_numpy(data: Union[bytes, np.ndarray]) -> np.ndarray:
    """ Convert to numpy array """
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    buffer = BytesIO(data)
    image = PImg.open(buffer)
    # Suppress: UserWarning: The given NumPy array is not writeable
    data = np.asarray(image)
    # Handle grayscale image
    if data.ndim == 2:
        data = np.tile(np.expand_dims(data, -1), (1, 1, 3))
    return data


def get_pair(data: Sequence[bytes],
             index: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Read image pair from array """
    idx = index * 2
    im1 = to_numpy(data[idx])
    im2 = to_numpy(data[idx + 1])
    return im1, im2


def default_transform(image: np.ndarray) -> np.ndarray:
    """ Convert to CHW, float32, [0 - 1]"""
    im = np.transpose(image.astype(np.float32), (2, 0, 1))
    return im / 255.0


def _to_numpy(group):
    if isinstance(group[0], (tuple, list)):
        for k, g in enumerate(group):
            group[k] = np.asarray(g)
        return group
    return np.asarray(group)


class NumpyBatcher(collections.abc.Iterable):
    """ Group iterable """

    def __init__(self,
                 iterable: Union[Iterable[Any], Sequence[Any]],
                 size: int = 16):
        self.iterable = iterable
        self.g_size = size

    def __iter__(self):
        iterator = iter(self.iterable)
        while True:
            group = []
            try:
                for _ in range(self.g_size):
                    elem = next(iterator)
                    if isinstance(elem, (tuple, list)):
                        if len(group) == 0:
                            group = [[e] for e in elem]
                        else:
                            for g, e in zip(group, elem):
                                g.append(e)
                    else:
                        group.append(elem)
                yield _to_numpy(group)
            except StopIteration:
                # Partially filled group
                if len(group) > 0:
                    yield _to_numpy(group)
                break


class VerificationDataset(collections.abc.Sequence):
    """
    Biometric verification dataset:
    """

    @staticmethod
    def load(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ Load verification data from a given file """
        # Load binary data
        obj = _read_bin(path)
        bins = obj[0]
        # Read images
        np_pairs = []
        n_pairs = len(bins) // 2
        for k in range(n_pairs):
            pair = get_pair(bins, k)
            np_pairs.append(pair)
        # Done
        return np_pairs

    def __init__(self,
                 root: str,
                 transform: Optional[TRANSFORM_FN] = None):
        # Default
        if transform is None:
            transform = default_transform
        # Load data
        self.pairs = self.load(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    # NOTE: Why we need @overload here
    # https://joshuacook.netlify.app/post/typehinting-list-subclass

    @overload
    def __getitem__(self, index: int) -> DATA_POINT:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[DATA_POINT]:
        ...

    def __getitem__(self,
                    index: Union[int, slice]) -> Union[DATA_POINT,
                                                       List[DATA_POINT]]:
        # Get corresponding elements
        pairs = self.pairs[index]
        # Apply transform if needed
        if self.transform is not None:
            if not isinstance(pairs, list):
                pairs = [pairs]
            # NOTE: why casting -> https://github.com/python/mypy/issues/5068
            pairs = [cast(DATA_POINT, tuple(self.transform(p) for p in pair))
                     for pair in pairs]
        if len(pairs) == 1:
            return pairs[0]
        return pairs
