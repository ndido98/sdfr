import argparse
from pathlib import Path
import os

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import graph_tool as gt
import graph_tool.topology as gtt
from tqdm import tqdm


def np_load_parallel(files: list[Path], n_jobs: int = 1) -> list[np.ndarray]:
    return joblib.Parallel(n_jobs=n_jobs, prefer="threads")(joblib.delayed(np.load)(f) for f in files)


def build_similarity_matrix(images: list[Path], metric: str = "cosine", n_jobs: int = 1) -> np.ndarray:
    embeddings = np_load_parallel([f.with_suffix(".npy") for f in images], n_jobs=n_jobs)
    embeddings = np.stack(embeddings)
    return pairwise_distances(embeddings, embeddings, metric=metric, n_jobs=n_jobs)


def build_subject_similarity_graph(
    subject_path: Path,
    threshold: float,
    metric: str = "cosine",
    n_jobs: int = 1,
) -> tuple[gt.Graph, gt.VertexPropertyMap]:
    images = sorted(subject_path.glob("*.png"))
    similarity_matrix = build_similarity_matrix(images, metric=metric, n_jobs=n_jobs)
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1] == len(images)
    adjacency_matrix = similarity_matrix <= threshold
    edges = np.array(np.nonzero(adjacency_matrix)).T
    edges = [(u, v) for u, v in edges if u < v]
    graph = gt.Graph(len(images), directed=False)
    graph.add_edge_list(edges)
    assert graph.num_vertices() == len(images), f"Graph for subject {subject_path} has {graph.num_vertices()} vertices, but there are {len(images)} images"
    images_prop = graph.new_vertex_property("string")
    for i, image in enumerate(images):
        images_prop[graph.vertex(i)] = image.stem
    return graph, images_prop


def get_most_similar_images(subject_path: Path, threshold: float, metric: str = "cosine", n_jobs: int = 1) -> list[Path]:
    graph, images_prop = build_subject_similarity_graph(subject_path, threshold, metric=metric, n_jobs=n_jobs)
    try:
        max_clique = max(gtt.max_cliques(graph), key=len)
    except ValueError:
        return []
    return sorted([subject_path / f"{images_prop[v]}.png" for v in max_clique])


def filter_dataset(
    dataset: Path,
    output: Path,
    min_images_per_class: int,
    distance_metric: str,
    distance_threshold: float,
    verbose: bool = False,
    n_jobs: int = 1,
) -> None:
    total_images, good_images = 0, 0
    good_classes = 0
    with open(output, "w") as f:
        dirs = [Path(d) for d in os.scandir(dataset) if d.is_dir()]
        total_classes = len(dirs)
        for class_path in tqdm(dirs):
            if verbose:
                tqdm.write(f"Processing {class_path}")
            class_total_images = len(list(class_path.glob("*.png")))
            total_images += class_total_images
            images = get_most_similar_images(class_path, distance_threshold, distance_metric, n_jobs=n_jobs)
            class_good_images = len(images)
            if class_good_images == 0:
                tqdm.write(f"Skipping class {class_path.name} because it has no good images")
                continue
            if class_good_images < min_images_per_class:
                tqdm.write(f"Skipping class {class_path.name} because it only has {class_good_images} images")
                continue
            good_classes += 1
            good_images += class_good_images
            if verbose:
                tqdm.write(f"Kept {class_good_images} out of {class_total_images} images for class {class_path.name}")
            f.write("\t".join(str(image.relative_to(dataset)) for image in images))
            f.write("\n")
    tqdm.write(f"Kept {good_images} out of {total_images} images, {good_classes} out of {total_classes} classes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-images-per-class", type=int, default=4)
    parser.add_argument("--distance-metric", type=str, default="euclidean")
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)

    args = parser.parse_args()

    filter_dataset(
        args.dataset,
        args.output,
        args.min_images_per_class,
        args.distance_metric,
        args.distance_threshold,
        args.verbose,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()