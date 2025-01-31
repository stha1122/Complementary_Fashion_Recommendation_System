import ast
import os

import sys
sys.path.append(r"C:/Users/Swetha/Desktop/Complete_the_Look_Recommendation_System")


import pandas as pd
from PIL import Image, UnidentifiedImageError
from src.config import config as cfg
from torch.utils.data import Dataset


# from PIL import Image, UnidentifiedImageError
# import os

class FashionProductSTLDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None, subset=None):
        self.image_dir = image_dir
        self.metadata = (
            pd.read_csv(metadata_file)
            if not subset
            else pd.read_csv(metadata_file)[pd.read_csv(metadata_file)["image_type"] == subset]
        )
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_id = self.metadata.iloc[index, 0]
        img_path = os.path.join(cfg.PACKAGE_ROOT, "dataset/", self.metadata.loc[img_id, "image_path"])

        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Error loading image at index {index} - Path: {img_path}. Exception: {e}")
            # Skip the problematic image and move to the next index
            return self.__getitem__(index + 1)

        if self.transform is not None:
            img = self.transform(img)

        return img

class FashionProductCTLTripletDataset(Dataset):
    def __init__(self, image_dir, metadata_file, data_type="train", transform=None):
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        triplet_id = self.metadata.reset_index().iloc[index, 0]
        data_src_folder = "train" if self.data_type in ["train", "validation"] else "test"
        img_triplets = []

        try:
            for img_type in ["anchor", "pos", "neg"]:
                img = Image.open(
                    os.path.join(
                        cfg.PACKAGE_ROOT,
                        "dataset/data/fashion_v2/",
                        f"{data_src_folder}_single",
                        self.metadata.loc[triplet_id, f"image_signature_{img_type}"]
                        + "_"
                        + self.metadata.loc[triplet_id, f"{img_type}_product_type"]
                        + ".jpg",
                    )
                ).convert("RGB")
                img_triplets.append(img)
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Error loading triplet at index {index}. Exception: {e}")
            # Handle as needed: skip the triplet or return placeholders

        if self.transform is not None:
            img_triplets = [self.transform(img) for img in img_triplets]

        return tuple(img_triplets)

class FashionProductCTLSingleDataset(Dataset):
    def __init__(self, image_dir, metadata_file, data_type="train", transform=None):
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.create_metadata(metadata_file)

    def create_metadata(self, metadata_file):
        metadata = pd.read_csv(metadata_file)
        self.metadata = metadata[metadata["image_type"] == self.data_type]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        self.metadata = self.metadata[self.metadata["image_type"] == self.data_type]
        img_id = self.metadata.reset_index().iloc[index, 0]

        try:
            img = Image.open(
                os.path.join(
                    cfg.PACKAGE_ROOT,
                    "dataset/data/fashion_v2/",
                    f"{self.data_type}_single",
                    self.metadata.loc[img_id, "image_single_signature"] + ".jpg",
                )
            ).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Error loading single image at index {index}. Exception: {e}")
            # Skip the image or return a placeholder or handle as needed
            return self.__getitem__(index + 1)  # Move to the next index or handle accordingly

        if self.transform is not None:
            img = self.transform(img)

        return img
