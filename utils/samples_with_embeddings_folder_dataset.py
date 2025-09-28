import os
from typing import Callable

import torch
import torchvision
from PIL import Image
from torchvision.datasets import DatasetFolder


class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFolderDataset(DatasetFolder):

    def __init__(self,
                 samples_root: str,
                 embeddings_root,
                 embedding_file_ending: str = '.npy',
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super(SamplesWithEmbeddingsFolderDataset, self).__init__(
            root=embeddings_root,
            loader=sample_loader,
            extensions=embedding_file_ending,
            transform=sample_transform,
            target_transform=None,
            is_valid_file=None
        )

        self.update_samples(sample_root=samples_root)

    def find_classes(self, directory):
        classes = [""]
        class_to_idx = {"": None}

        return classes, class_to_idx

    def update_samples(self, sample_root: str):
        updated_samples = []
        for embedding_path, _ in self.samples:
            embeddings = torch.load(embedding_path)

            for image_name, id_context in embeddings.items():
                # print(image_name)
                image_path = os.path.join(sample_root, image_name)
                assert os.path.isfile(image_path)

                updated_samples.append((image_path, torch.from_numpy(id_context)))
        self.samples = updated_samples

    def __getitem__(self, index: int):
        image_path, id_context = self.samples[index]

        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, id_context


if __name__ == "__main__":
    # Assume the dataset is stored in "samples_path" and "embeddings_path"
    samples_path = '/home/moon/yy/newAID/data/redpanda'
    embeddings_path = '/home/moon/yy/newAID/data/redpanda_embeddings_clip'

    # Initialize the dataset
    dataset = SamplesWithEmbeddingsFolderDataset(samples_root=samples_path,
                                                 embeddings_root=embeddings_path,
                                                 embedding_file_ending='.npy',
                                                 sample_loader=PILImageLoader(),
                                                 sample_transform=torchvision.transforms.Compose([torchvision.transforms.Resize(size=(448, 448)),
                                                                                                  torchvision.transforms.ToTensor(),
                                                                                                  torchvision.transforms.RandomHorizontalFlip(p=0.5)])
                                                 )

    # Iterate over the first 5 items in the dataset and print their shapes/types
    print(len(dataset))
    for i in range(len(dataset)):
        sample, id_context = dataset[i]
        print(f"Sample {i} shape: {sample.shape}, ID_Context shape: {id_context.shape}")
