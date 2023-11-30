from glob import glob
from os.path import join
import cv2
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import os


def extract_embeddings(dataset_path, embeddings_path):
    from deepface import DeepFace

    folders = glob(join(dataset_path, '*'))

    for folder in tqdm(folders):
        images = glob(join(folder, '*.png'))

        # create the directory for saving the embedding
        if not os.path.exists(folder.replace(dataset_path, embeddings_path)):
            os.makedirs(folder.replace(dataset_path, embeddings_path))

        for image in images:

            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # the embedding is always extracted, even if there is not a face in the image
            embedding = DeepFace.represent(img, model_name="ArcFace", enforce_detection=False, detector_backend="mtcnn")
            embedding = embedding[0]["embedding"]

            file_name = image.replace(dataset_path, embeddings_path)[:-4]

            # save the embedding
            with open(file_name + '.npy', 'wb') as f:
                np.save(f, embedding)


# run this method only when embeddings are extractd
def compute_similarity(embedding_path, output_file_name, distance_theshold):

    # this file contains all the images that do not satisfy the identity check
    file = open(output_file_name, 'w')

    folders = sorted(glob(join(embedding_path, '*')))

    for folder in tqdm(folders):

        embeddings = sorted(glob(join(folder, '*.npy')))

        # find the best reference image (image with the lowest distance)
        embds = []
        for embedding in embeddings:
            embds.append(np.load(embedding))
        cm = distance.cdist(embds, embds, 'cosine')
        max_count = 0
        best_line = -1
        for i, row in enumerate(cm):
            count_under_threshold = sum(value < distance_theshold for value in row)
            if count_under_threshold > max_count:
                max_count = count_under_threshold
                best_line = i

        # take the embedding as reference
        reference_embedding = np.load(embeddings[best_line])

        for embedding in embeddings:
            emb = np.load(embedding)

            if distance.cosine(reference_embedding, emb) > distance_theshold:
                file.write(embedding + '\n')

    file.close()


if __name__ == "__main__":

    dataset_path = r'E:\StableDiffusion'
    embedding_path = r'E:\embeddings'
    output_file_name = 'images_with_low_similarity'

    distance_theshold = 0.4

    extract_embeddings(dataset_path, embedding_path)
    compute_similarity(embedding_path, output_file_name, distance_theshold)