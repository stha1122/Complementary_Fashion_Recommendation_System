import pickle
import torch
import sys
import os
sys.path.append(r"C:/Users/Swetha/Desktop/Complete_the_Look_Recommendation_System")


from torch import nn
from torchvision import models, transforms
from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader, FashionProductSTLDataloader
from src.models.Model import CompatibilityModel
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyleEmbedding:
    """
    Feature extractor that generates different features according to data_loader and task
    """

    def __init__(self):
        # Create the directory for cached embeddings if it doesn't exist
        os.makedirs(f"{cfg.PACKAGE_ROOT}/features/cached_embeddings", exist_ok=True)

    def similar_product_embedding(self, data_loader, task_name="similar_product"):
        """
        Import data loader with the batches. Go through each batch and pass through ResNet. Features are extracted from
        the last layer for each image in each batch. At the very end each batch is stacked and you are left with tensor
        "all_features" of shape (batches,batch_size,features)
        """

        # get pretrain model and remove the classification layer
        print("You are using device: %s" % device)
        resnet = models.resnet18(pretrained=True).to(device)
        resnet.fc = nn.Identity()
        resnet.eval()
        transform = transforms.Resize(256)
        all_features = []

        for batch in tqdm.tqdm(data_loader):
            X = transform(batch)  # resizes to 256 X 256 for ResNet
            X = X.float().to(device)
            with torch.no_grad():
                batch_features = resnet(X)
                all_features.append(batch_features)

        all_features = torch.cat(all_features).to("cpu")

        # save all features to a pickle file
        with open(
            f"{cfg.PACKAGE_ROOT}/features/cached_embeddings/{task_name}_embedding.pickle", "wb"
        ) as handle:
            pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_features

    def compatible_product_embedding(self, data_loader, task_name):
        """
        Import data loader with the batches. Go through each batch and pass through ComaptibilityModel. Features are extracted from
        the last layer for each image in each batch. At the very end each batch is stacked and you are left with tensor
        "all_features" of shape (batches,batch_size,features)
        """

        # get pretrain model and remove the classification layer
        print("You are using device: %s" % device)
        model = CompatibilityModel()
        model.load_state_dict(
            torch.load(f"{cfg.TRAINED_MODEL_DIR}/trained_compatibility_model_epoch5.pth", map_location=device)[
                "model_state_dict"
            ]
        )
        model.to(device)
        model.eval()
        transform = transforms.Resize(256)
        all_features = []

        for batch in tqdm.tqdm(data_loader):

            X = transform(batch)  # resizes to 256 X 256 for ResNet
            X = X.float().to(device)
            with torch.no_grad():
                batch_features = model(X)
                all_features.append(batch_features)

        all_features = torch.cat(all_features).to(
            "cpu"
        )  # send it to cpu for the pickle to be read properly

        # save all features to a pickle file
        with open(
            f"{cfg.PACKAGE_ROOT}/features/cached_embeddings/{task_name}_embedding.pickle", "wb"
        ) as handle:
            pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_features


if __name__ == "__main__":
    # StyleEmbedding().similar_product_embedding(
    #     data_loader=FashionProductSTLDataloader().data_loader(), task_name="similar_product"
    # )
    # StyleEmbedding().similar_product_embedding(
    #     data_loader=FashionCompleteTheLookDataloader().single_data_loader(),
    #     task_name="similar_prod_CTL",
    # )

    StyleEmbedding().compatible_product_embedding(
        data_loader=FashionCompleteTheLookDataloader(image_type="test").single_data_loader(),
        task_name="compatible_product_test",
    )

    StyleEmbedding().compatible_product_embedding(
        data_loader=FashionCompleteTheLookDataloader().single_data_loader(),
        task_name="compatible_product",
    )

