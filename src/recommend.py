import pickle
import sys
sys.path.append(r"C:/Users/Swetha/Desktop/Complete_the_Look_Recommendation_System")



import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.image_utils import display_compatible_images

from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader, FashionProductSTLDataloader
from src.utils.similarity import calculate_similarity



# Function to display images in a Tkinter window
def display_images(images):
    root = tk.Tk()
    root.title("Recommended Products")

    rows = 1
    columns = len(images)
    for i, image_path in enumerate(images):
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)

        label = tk.Label(root, image=img)
        label.image = img
        label.grid(row=0, column=i)

    root.mainloop()

    

def recommend_similar_products(product_id, task_name="similar_product", top_n=5):
    """takes in the product id and returns the top 5 similar product to the input product"""
    # get extracted features
    with (
        open(f"{cfg.PACKAGE_ROOT}/features/cached_embeddings/{task_name}_embedding.pickle", "rb")
    ) as file:
        all_similar_products_features = pickle.load(file)

    # get dataset metadata dataframe
    data_loader = FashionProductSTLDataloader().data_loader()
    metadata = data_loader.dataset.metadata

    # get query feature from product id
    product_metadata = metadata[metadata["product_id"] == product_id].to_dict(orient="records")[0]
    product_feature_vec = all_similar_products_features[product_id, :]

    # calculate similarities and get all of the 5 products metadata
    simlarity_score = calculate_similarity(
        product_feature_vec, all_similar_products_features, "cosine"
    )

    # get query product category and filter the prpduct catelog for the same category
    product_category = product_metadata["product_type"]
    metadata["similarity_score"] = simlarity_score.cpu()

    # get top 5 products metadata
    recommended_products_metadata = (
        metadata[
            (metadata["product_type"] == product_category) & (metadata["similarity_score"] != 1)
        ]
        .sort_values(by="similarity_score", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    return {
        "input_product": product_metadata,
        "recommended_products": recommended_products_metadata,
    }


def recommend_complementary_products(product_id, task_name="compatible_product", top_n=5):
    """takes in the product id and returns the top 5 compatible product to the input product"""
    # get extracted features
    with (
        open(f"{cfg.PACKAGE_ROOT}/features/cached_embeddings/{task_name}_embedding.pickle", "rb")
    ) as file:
        all_compatible_products_features = pickle.load(file)

    # get dataset metadata dataframe
    data_loader = FashionCompleteTheLookDataloader().single_data_loader()
    metadata = data_loader.dataset.metadata[
        ["product_id", "image_single_signature", "product_type", "image_path"]
    ]

    # get query feature from product id
    product_metadata = metadata[metadata["product_id"] == product_id].to_dict(orient="records")[0]
    product_feature_vec = all_compatible_products_features[product_id, :]

    # calculate compatibility score and get all of the 5 products metadata
    compatibility_score = calculate_similarity(
        product_feature_vec, all_compatible_products_features, "cosine"
    )

    # get query product category and filter the product catelog for the same category
    input_product_category = product_metadata["product_type"]
    metadata["compatibility_score"] = compatibility_score.cpu()

    # get top 5 products metadata
    recommended_products_metadata_all_cat = (
        metadata[(metadata["product_type"] != input_product_category)]
        .sort_values(by="compatibility_score", ascending=False)
        .groupby("product_type")
        .head(1)
    )

    return {
        "input_product": product_metadata,
        "recommended_compatible_products": recommended_products_metadata_all_cat.sort_values(
            by="compatibility_score", ascending=False
        )
        .head(top_n)
        .to_dict(orient="records"),
    }


# if __name__ == "__main__":
#     import random

#     # similar_recommendations = recommend_similar_products(product_id=random.randint(1, 38000))
#     compatible_recommendations = recommend_complementary_products(
#         product_id=random.randint(1, 454000)
#     )

#     from utils.image_utils import display_compatible_images, display_recommended_products

#     # print(similar_recommendations)
#     # display_recommended_products(
#     #     similar_recommendations["input_product"]["image_path"],
#     #     *[rec["image_path"] for rec in similar_recommendations["recommended_products"]],
#     #     [
#     #         round(rec["similarity_score"], 3)
#     #         for rec in similar_recommendations["recommended_products"]
#     #     ],
#     # )

#     print(compatible_recommendations)
#     # display_recommended_products(
#     #     compatible_recommendations["input_product"]["image_path"],
#     #     *[
#     #         rec["image_path"]
#     #         for rec in compatible_recommendations["recommended_compatible_products"]
#     #     ],
#     #     [
#     #         round(rec["compatibility_score"], 3)
#     #         for rec in compatible_recommendations["recommended_compatible_products"]
#     #     ],
#     #     save_image=True,
#     # )
#     display_compatible_images(
#         compatible_recommendations["input_product"]["image_single_signature"] + ".jpg",
#         *[
#             rec["image_single_signature"] + ".jpg"
#             for rec in compatible_recommendations["recommended_compatible_products"]
#         ],
#         product_id=compatible_recommendations["input_product"]["product_id"],
#         save_image=True,
#     )



#      # Generate random product_id
#     product_id = random.randint(1, 454000)

#     # Get recommendations
#     compatible_recommendations = recommend_complementary_products(product_id)

#     # Extract image paths
#     input_image_path = os.path.join(cfg.DATASET_DIR, "data/fashion_v2/train_single", f"{compatible_recommendations['input_product']['image_single_signature']}.jpg")
#     recommendation_paths = [
#         os.path.join(cfg.DATASET_DIR, "data/fashion_v2/train_single", f"{rec['image_single_signature']}.jpg")
#         for rec in compatible_recommendations['recommended_compatible_products']
#     ]

#     # Display the images using Tkinter
#     display_images([input_image_path] + recommendation_paths)



import tkinter as tk
from PIL import Image, ImageTk

class RecommendationApp:
    def __init__(self, input_image_path, recommendation_data):
        self.root = tk.Tk()
        self.root.title("Product Recommendations")

        # Load and display the input image
        self.input_img = Image.open(input_image_path)
        self.input_img = ImageTk.PhotoImage(self.input_img)
        self.input_label = tk.Label(self.root, image=self.input_img, text="Input Image", compound="top")
        self.input_label.grid(row=0, column=0)

        # Load and display the recommendation images with labels
        self.recommendation_images = []
        for i, rec in enumerate(recommendation_data):
            img_path = rec['image_path']
            img = Image.open(img_path)
            img_tk = ImageTk.PhotoImage(img)
            self.recommendation_images.append(img_tk)
            label = tk.Label(self.root, image=img_tk, text=f"Type: {rec['product_type']}\nCompatibility: {rec['compatibility_score']:.4f}", compound="top")
            label.grid(row=0, column=i + 1)

    def run(self):
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    import random

    # Generate random product_id
    product_id = random.randint(1, 454000)

    # Get recommendations
    compatible_recommendations = recommend_complementary_products(product_id)

    # Extract image paths and compatibility scores
    input_image_path = os.path.join(cfg.DATASET_DIR, "data/fashion_v2/train_single", f"{compatible_recommendations['input_product']['image_single_signature']}.jpg")
    recommendation_data = [
        {
            'image_path': os.path.join(cfg.DATASET_DIR, "data/fashion_v2/train_single", f"{rec['image_single_signature']}.jpg"),
            'product_type': rec['product_type'],
            'compatibility_score': rec['compatibility_score'],
        }
        for rec in compatible_recommendations['recommended_compatible_products']
    ]

    # Display the images using Tkinter
    app = RecommendationApp(input_image_path, recommendation_data[:4])
    app.run()
