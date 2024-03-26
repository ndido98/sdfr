import argparse
from pathlib import Path
import multiprocessing as mp
import time
import os
import gc
import random

import numpy as np
import cv2
from tqdm import tqdm


# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_all_images(path: Path) -> list[str]:
    return sorted(path.rglob("*.png"))


def compute_embedding(image_path: Path) -> None:
    attempts = 0
    if image_path.with_suffix(".npy").exists():
        return
    while True:
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255
            break
        except:
            wait = 100 * (2 ** attempts)
            print(f"Failed to load {image_path}, retrying in {wait} milliseconds")
            time.sleep(wait / 1000)
            attempts += 1
            if attempts >= 10:
                raise
    import tensorflow as tf
    from deepface import DeepFace
    with tf.device("/CPU:0"):
        embedding = DeepFace.represent(image, model_name="ArcFace", enforce_detection=False, detector_backend="skip")
        embedding = np.array(embedding[0]["embedding"].copy(), dtype=np.float64).squeeze().copy()
    del image
    attempts = 0
    while True:
        try:
            with open(image_path.with_suffix(".npy"), "wb") as f:
                np.save(f, embedding)
            break
        except:
            wait = 100 * (2 ** attempts)
            print(f"Failed to save {image_path.with_suffix('.npy')}, retrying in {wait} milliseconds")
            time.sleep(wait / 1000)
            attempts += 1
            if attempts >= 10:
                raise
    del embedding


def process_image(image_path: Path) -> None:
    compute_embedding(image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.files:
        images = args.files
    elif args.dataset:
        images = get_all_images(args.dataset)
    else:
        raise ValueError("Either files or --dataset must be specified")
    if args.num_workers == 0:
        for image in tqdm(images):
            process_image(image)
    else:
        with mp.Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_image, images), total=len(images)):
                pass


if __name__ == "__main__":
    main()