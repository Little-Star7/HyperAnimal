import os
from typing import Any
import hydra
import argparse
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf, DictConfig
import numpy as np
from utils.helpers import ensure_path_join

from tqdm import tqdm

import sys


def load_image_paths(datadir, skip_files=None):
    img_files = os.listdir(datadir)

    if skip_files is not None:
        N = len(img_files)
        img_files = list(filter(lambda fname: fname not in skip_files, img_files))
        print(f"Skipped {N - len(img_files)} out of {N} images due to skip_files of length {len(skip_files)}")

    return [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".png") or f_name.endswith(".jpg")]


class ImageSplitInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, skip_files=None, image_size=512):

        self.img_paths = load_image_paths(datadir, skip_files)
        # print("Number of images:", len(self.img_paths))

        self.image_size = image_size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = image.convert("RGB")

        image = torchvision.transforms.functional.to_tensor(image)

        nrows = image.shape[1] // self.image_size
        ncols = image.shape[2] // self.image_size

        id_images = []
        for r in range(nrows):
            for c in range(ncols):
                tile = image[:, r * self.image_size: (r + 1) * self.image_size,
                       c * self.image_size: (c + 1) * self.image_size]
                id_images.append(tile)

        return torch.stack(id_images), os.path.basename(img_path)

    def __len__(self):
        return len(self.img_paths)


def main(samples_dir):
    dataset = ImageSplitInferenceDataset(datadir=samples_dir, skip_files=None, image_size=512)

    for i, (id_images, id_name) in tqdm(enumerate(dataset), total=len(dataset)):
        id_name = id_name.split(".")[0]
        for i, img in enumerate(id_images):
            save_dir = os.path.join(samples_dir, "dataset", id_name)
            os.makedirs(save_dir, exist_ok=True)

            save_image(img, os.path.join(save_dir, f"{id_name}_{i:02}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--samples_dir", type=str, help="Directory containing the sample images", required=True)
    # parser.add_argument("--samples_dir", type=str, help="Directory containing the sample images", default="samples/default_dir")
    args = parser.parse_args()

    main(args.samples_dir)


# if __name__ == "__main__":
#
#     samples_dir = "samples/rp_f1/syn_2000"
#     dataset = ImageSplitInferenceDataset(datadir=samples_dir, skip_files=None, image_size=512)
#
#     for i, (id_images, id_name) in tqdm(enumerate(dataset), total=len(dataset)):
#         id_name = id_name.split(".")[0]
#         for i, img in enumerate(id_images):
#             save_dir = os.path.join(samples_dir, "dataset", id_name)
#             os.makedirs(save_dir, exist_ok=True)
#
#             save_image(img, os.path.join(save_dir, f"{id_name}_{i:02}.png"))
