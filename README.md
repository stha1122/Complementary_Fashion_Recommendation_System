# Fashion_Recommendation_System

## Introduction

With the accelerated online eCommerce scene driven by the contactless shopping style in recent years, having a great recommendation system is essential to the business' success. However, it has always been challenging to provide any meaningful recommendations with the absence of user interaction history, known as the cold start problem.  In this project, we attempted to create a comprehensive recommendation system that recommends both similar and complementary products using the power of deep learning and visual embeddings, which would effectively recommend products without need any knowledge of user preferences, user history, item propensity, or any other data.


## Datasets

The dataset used is the [shop the look dataset](https://github.com/kang205/STL-Dataset) and the [complete the look dataset](https://github.com/eileenforwhat/complete-the-look-dataset) from Pinterest. Thank you for kindly sharing these great data sources to make this project possible.

## Quick Run Instructions

#### Recommend Similar Products
1. Download data - Run `cd src/dataset/data` and run `python download_data.py`
2. Get similar product embedding - Run `cd src`, make sure the in the `features/Embedding.py`, the class method `similar_product_embedding` is being selected, then run ` PYTHONPATH=../:. python features/Embedding.py` (be careful this could take up to 2 hours without the presence of a GPU)
3. Recommend Similar Product - Run `cd src`, make sure the in the `recommend.py`, the function `recommend_similar_products` is being selected, then run ` PYTHONPATH=../:. python recommend.py`

#### Recommend Compatible Products
1. Download data - Run `cd src/dataset/data` and run `python download_data.py`
2. Train compatible model - Run `cd src` and run ` PYTHONPATH=../:. python models/training.py`
3. Get compatible product embedding - Run `cd src`, make sure the in the `features/Embedding.py`, the class method `compatible_product_embedding` is being selected, then run ` PYTHONPATH=../:. python features/Embedding.py` (be careful this could take up to 15 hours without the presence of a GPU, 7 hours with GPU)<img 
4. Evaluate the compatible model -  Run `cd src` and run ` PYTHONPATH=../:. python models/evaluate.py`
5. Recommend Compatible Product - Run `cd src`, make sure the in the `recommend.py`, the function `recommend_compatible_products` is being selected, then run ` PYTHONPATH=../:. python recommend.py`

## Results
<img width="903" height="552" alt="image" src="https://github.com/user-attachments/assets/752210ab-53f0-4740-9b9d-b4c82f86abc0" />
<img width="903" height="505" alt="image" src="https://github.com/user-attachments/assets/33169f00-e8bb-4550-8379-27a177f21303" />
<img width="957" height="457" alt="image" src="https://github.com/user-attachments/assets/d4fc4488-7230-4cbd-94bf-eb4645004ed4" />
<img width="1022" height="488" alt="image" src="https://github.com/user-attachments/assets/deb9a4a4-f046-44a5-a29e-e7dbcb2a7951" />
<img width="781" height="464" alt="image" src="https://github.com/user-attachments/assets/4e65e177-6d65-4ba7-9760-02846d6c6323" />
<img width="903" height="502" alt="image" src="https://github.com/user-attachments/assets/2be12cb9-74cb-4e20-8dd8-ceae74dc38a9" />
<img width="903" height="500" alt="image" src="https://github.com/user-attachments/assets/2a91beb3-214c-4fd3-a33a-780ff9c3ed29" />

https://github.com/user-attachments/assets/cd10177b-1046-4ed6-9503-5c17f8b7aa19


## Credits

This project is based on [Complete_the_Look_Recommendation_System](https://github.com/chen-bowen/Complete_the_Look_Recommendation_System).

