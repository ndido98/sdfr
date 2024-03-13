import argparse
from pathlib import Path
import multiprocessing as mp
import os

import numpy as np
import cv2
from facenet_pytorch import MTCNN
from align_utils import warp_and_crop_face, get_reference_facial_points
from tqdm import tqdm


def get_all_images(path: Path) -> list[str]:
    files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith((".png", ".jpg")):
                files.append(Path(entry.path))
            elif entry.is_dir():
                files.extend(get_all_images(entry.path))
    return sorted(files)


def load_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] < 112 or img.shape[1] < 112:
        factor = 224 / min(img.shape[0], img.shape[1])
        img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return img


def detect_face_batch(images: list[np.ndarray], detector: MTCNN) -> list[np.ndarray | None]:
    boxes, probs, landmarks = detector.detect(images, landmarks=True)
    # Choose the face with highest probability in each image of the batch
    faces = []
    for i in range(len(images)):
        if boxes[i] is None:
            faces.append(None)
            continue
        highest_prob = np.argmax(probs[i])
        landmark = landmarks[i][highest_prob]
        faces.append(landmark)
    return faces


def align_face(image: np.ndarray, landmarks: np.ndarray | None, image_size: int) -> np.ndarray | None:
    reference_points = get_reference_facial_points(default_square=True)
    scaled_reference_points = reference_points * (image_size / 112)
    if landmarks is not None:
        return warp_and_crop_face(image, landmarks, reference_pts=scaled_reference_points, crop_size=(image_size, image_size))
    else:
        return None


def save_image(image: np.ndarray | None, input_root: Path, output_root: Path, image_path: Path) -> None:
    if image is None:
        print(f"Image {image_path} does not contain a face")
        return
    relative_image_path = image_path.relative_to(input_root)
    image_output_path = output_root / relative_image_path
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_output_path), image)


def process_image(args: tuple[Path, Path, Path, str, int]) -> None:
    image_path, dataset_path, output_path, device, image_size = args
    if (output_path / image_path.relative_to(dataset_path)).exists():
        return
    image = load_image(image_path)
    pts_file = image_path.with_suffix(".pts")
    if pts_file.exists():
        # Found a 68 landmarks file; get right eye, left eye, nosetip, right mouth corner, left mouth corner
        pts = pts_file.read_text().splitlines()[3:-1]
        landmarks = np.array([[float(x) for x in pt.split()] for pt in pts])
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        nosetip = landmarks[30]
        right_mouth = landmarks[48]
        left_mouth = landmarks[54]
        landmarks = np.array([right_eye_center, left_eye_center, nosetip, right_mouth, left_mouth])
    else:
        detector = MTCNN(keep_all=True, device=device)
        landmarks = detect_face_batch([image], detector)[0]
    aligned_face = align_face(image, landmarks, image_size)
    save_image(aligned_face, dataset_path, output_path, image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    batch_size = args.batch_size
    image_size = args.image_size

    detector = MTCNN(thresholds=[0.1, 0.2, 0.5], factor=0.95, keep_all=True, device=args.device)

    all_images = get_all_images(dataset_path)
    print(f"Found {len(all_images)} images")

    with mp.Pool(args.num_workers) as pool:
        for i in tqdm(range(0, len(all_images), batch_size)):
            images_paths = all_images[i : i + batch_size]
            output_paths = [output_path / image_path.relative_to(dataset_path) for image_path in images_paths]
            images_paths = [image_path for image_path, output_path in zip(images_paths, output_paths) if not output_path.exists()]
            if len(images_paths) == 0:
                continue
            images = pool.map(load_image, images_paths)
            images_landmarks = detect_face_batch(images, detector)
            align_face_args = list(zip(images, images_landmarks, [image_size] * len(images)))
            aligned_faces = pool.starmap(align_face, align_face_args)
            save_image_args = list(zip(aligned_faces, [dataset_path] * len(images), [output_path] * len(images), images_paths))
            pool.starmap(save_image, save_image_args)


if __name__ == "__main__":
    main()
