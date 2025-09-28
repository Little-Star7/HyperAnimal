import os
import argparse
from tqdm import tqdm
import sys

import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np

from models.identification.identification_model import ft_net
import torchvision
import inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parent_dir)
import sys
sys.path.insert(0, 'IDiff-Animal/')


def load_img_paths(datadir):
    """load animal images"""
    img_files = sorted(os.listdir(datadir))
    return [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".jpg") or f_name.endswith(".png")]


class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.img_paths = load_img_paths(datadir)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 1

    print("Dataset:", args.data_dir)
    if args.model_name == "resnet50":
        print("loading resnet50 identification model...")
        model = ft_net(class_num=107, linear_num=512)
        ckpt = torch.load("models/identification/weights/atrw.pth", map_location=device)
        model.load_state_dict(ckpt)
    else:
        raise NotImplementedError

    model.to(device)
    model.eval()
    dataset = ImageInferenceDataset(args.data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting encoding ...")
    content = {}
    for i, (img_batch, filename_batch) in tqdm(enumerate(loader), total=len(loader)):
        img_batch = torchvision.transforms.functional.resize(img_batch, 448)
        img_batch = img_batch.to(device)
        emb_batch = model(img_batch)
        emb_batch = torch.nn.functional.normalize(emb_batch).detach().cpu().numpy()

        for emb, filename in zip(emb_batch, filename_batch):
            content[filename] = emb

    # print(content)
    torch.save(content, os.path.join(args.out_dir, f"embeddings_{args.model_name}_" + os.path.basename(args.data_dir) + ".npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="[resnet50, convnext]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/atrw",
        help="path to data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/atrw_embeddings",
        help="directory to save embedding file to"
    )
    args = parser.parse_args()
    main(args)
