# Incorporating Item Semantics into User and Item Latent Factor Models for Recommender System
Explainable Recommendations with Reinforcement Learning

## Dependencies
The code was run using Python *3.10.9*. All python dependencies are stored in **dependencies.txt**, which can be installed using the following command.

```
python3 -m pip install -r dependencies.txt
```

## Dataset
The dataset we have chosen is Amazon's 2014 review dataset hosted by the University of California San Diego. The Amazon datasets have been preprocessed to remove any duplicate items as well as removing all users and items with less than 5 reviews associated with them. We chose these 3 subsets of the Amazon dataset in order to observe how the model performs with a variety of both similar *(Toys vs Android)* and different *(Toys/Android vs Health)* categories.

*[Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)*

1. **Toys & Games**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz)*
   - *[Full Reviews Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games.json.gz)*

2. **Apps for Android**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz)*
   - *[Full Reviews Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android.json.gz)*

3. **Health & Personal Care**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz)*
   - *[Full Reviews Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care.json.gz)*