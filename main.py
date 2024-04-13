import torch
import pickle
import os
import streamlit as st
from torchvision import transforms
from PIL import Image
from src.models.Model import CompatibilityModel
from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
import sys
sys.path.append(r"C:/Users/Swetha/Desktop/Complete_the_Look_Recommendation_System/src")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import streamlit as st
import os
from PIL import Image
from src.config import config as cfg
from recommend import recommend_complementary_products
import pandas as pd

# Function to resize and display images horizontally in multiple rows
def display_horizontal_images(images, captions, row_length=4, width=100, height=100, space_between_images=20):
    cols = st.columns(row_length)
    for i, (image, caption) in enumerate(zip(images, captions)):
        try:
            img = Image.open(image)
            img_resized = img.resize((width, height))
            with cols[i % row_length].container():
                st.image(img_resized,use_column_width=True)
            # if (i + 1) % row_length != 0:
            #     st.write("", width=space_between_images)
        except FileNotFoundError:
            cols[i % row_length].write(f"Unable to display image: {caption}")



def main():
    st.title('Complementary Product Recommender')

    # Load the metadata to get clothing categories
    metadata_file = os.path.join(cfg.DATASET_DIR, "metadata", "dataset_metadata_ctl_single.csv")
    metadata = pd.read_csv(metadata_file)
    categories = metadata['product_type'].unique()

    # Dropdown to select clothing category
    selected_category = st.selectbox('Select a clothing category:', categories)

    if selected_category:
        # Filter metadata based on selected category
        category_metadata = metadata[metadata['product_type'] == selected_category]

        # Display at most 20 clothing items horizontally
        num_items = min(len(category_metadata), 20)
        images = [os.path.join(cfg.DATASET_DIR, row['image_path']) for _, row in category_metadata.head(num_items).iterrows()]
        captions = category_metadata['image_single_signature'].tolist()[:num_items]
        selected_product_id = None

        cols = st.columns(5)  # Create 5 columns for displaying images
        for i, (image, caption) in enumerate(zip(images, captions)):
            try:
                img = Image.open(image)
                img_resized = img.resize((100, 100))  # Resize input image to match output size
                cols[i % 5].image(img_resized, use_column_width=True)
                button_key = f"select_button_{i}"

                if cols[i % 5].button("Select", key=button_key):
                    selected_product_id = category_metadata.iloc[i]['product_id']

            except FileNotFoundError:
                cols[i % 5].write(f"Unable to display image: {caption}")

        if selected_product_id:
            # Recommend complementary products based on the selected clothing item
            recommendations = recommend_complementary_products(selected_product_id)

            # Display the input clothing item on the left
            st.header('Input Product')
            selected_cloth_image_path = os.path.join(cfg.DATASET_DIR, category_metadata.loc[category_metadata['product_id'] == selected_product_id, 'image_path'].iloc[0])
            selected_cloth_image = Image.open(selected_cloth_image_path)
            input_img_resized = selected_cloth_image.resize((200, 200))  # Resize input image to match output size
            st.image(input_img_resized, use_column_width=False)



            # # Add space between input and output
            # st.write("")

            # Display the recommended products on the right
            st.header('Recommended Products')
            recommended_images = [os.path.join(cfg.DATASET_DIR, rec['image_path']) for rec in recommendations['recommended_compatible_products']]
            recommended_captions = [rec['image_single_signature'] for rec in recommendations['recommended_compatible_products']]
            display_horizontal_images(recommended_images, recommended_captions)

if __name__ == "__main__":
    main()
