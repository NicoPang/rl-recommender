# rl-recommender
Explainable Recommendations with Reinforcement Learning

# Datasets
Datasets typically exceed Github's individual file size limit of 100 MB, so they will have to be manually downlaoded each time. Unzip them into **data/datasets** to work with the existing code. The datasets are as follows:
1. https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- This is Amazon's 1996-2014 Product Data. We _specifically_ use the **Toys and Games 5-Core** file, which downloads as a .gz. It should have 167,597 reviews.
2. https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
- This is Yelp's 2018 challenge dataset. Unfortunately, this no longer appears to be available on the Yelp Website, but Yelp provides a copy of the dataset on Kaggle. Using this dataset is a bit tricky; there is a lot of data from all types of establishments throughout the US, but the paper solely deals with restaurants in Las Vegas, NV. Unfortunately, for some reason there do not appear to be any businesses within Las Vegas, NV. More time will have to be spent looking into this.

