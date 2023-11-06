from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd


# Sum the reviews into 1 document and then do LDA to get topics
def sum_reviews_LDA(df: pd.DataFrame):
    grouped_review_df = (
        df.groupby("Item_ID")["Review_Body"].apply(lambda x: " ".join(x)).reset_index()
    )
    grouped_review_df = grouped_review_df.reset_index(drop=True)
    grouped_review_df = grouped_review_df.sort_values(by="Item_ID")
    Item_ID_list = grouped_review_df["Review_Body"].tolist()

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#examples-using-sklearn-feature-extraction-text-countvectorizer

    vectorizer = TfidfVectorizer(encoding="utf-8", lowercase=True)
    X = vectorizer.fit_transform(Item_ID_list)
    topics_n = 10
    top_n_words = 30

    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=topics_n, random_state=42, n_jobs=-1)
    lda.fit(X)
    document_topics = lda.transform(X)
    tf_feature_names = vectorizer.get_feature_names_out()

    top_words_per_topic = []
    for topic in lda.components_:
        top_word_indices = topic.argsort()[-top_n_words:][::-1]
        top_words = [tf_feature_names[i] for i in top_word_indices]
        top_words_per_topic.append(top_words)

    for i, words in enumerate(top_words_per_topic):
        print(f"Topic {i + 1}: {', '.join(words)}")

    df = pd.DataFrame(document_topics)
    df.index.name = "Item_ID"

    return df


def user_id_topics_avg(df_Item_topics: pd.DataFrame, df_user_item: pd.DataFrame):
    df_user_item = df_user_item[["User_ID", "Item_ID"]]
    merged_df = df_user_item.merge(df_Item_topics, on="Item_ID")
    user_avg_features = merged_df.groupby("User_ID").mean().reset_index()
    user_avg_features.drop("Item_ID", axis=1, inplace=True)
    return user_avg_features


if __name__ == "__main__":
    import os

    subsets = {
        "Toys_and_Games_5.json",
        "reviews_Toys_and_Games.json",
        "Apps_for_Android_5.json",
        "reviews_Apps_for_Android.json",
        "Health_and_Personal_Care_5.json",
        "reviews_Health_and_Personal_Care.json",
    }
    total_df = pd.DataFrame(
        columns=["User_ID", "Item_ID", "Review_Body", "Review_Score", "Review_Header"]
    )

    # (1) Check if datasets on system
    git_path = os.path.abspath(__file__)[
        :-16
    ]  # make sure not to change current filename or folder names
    for name in subsets:
        if "5" not in name:
            continue
        p = os.path.join(git_path, "datasets", "raw", name)
        if os.path.exists(path=p) is False:
            print(f"ERROR: issue not finding unzipped file at {p}")

        # (2) Combine the unzipped json datasets
        df = pd.read_json(path_or_buf=p, lines=True)

        df = df.drop(
            ["reviewerName", "helpful", "unixReviewTime", "reviewTime"], axis=1
        )
        df = df.rename(
            columns={
                "reviewerID": "User_ID",
                "asin": "Item_ID",
                "reviewText": "Review_Body",
                "summary": "Review_Header",
                "overall": "Review_Score",
            }
        )
        total_df = pd.concat([df, total_df], axis=0)
        break

    # (3) Create a mapping so we don't have to deal with random hashes for User_ID and Item_ID
    user_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["User_ID"].unique())
    }
    total_df["User_ID"] = total_df["User_ID"].map(user_id_mapping)

    item_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["Item_ID"].unique())
    }
    total_df["Item_ID"] = total_df["Item_ID"].map(item_id_mapping)

    # (4) Testing LDA
    x = sum_reviews_LDA(total_df)
    """
    total_df = pd.merge(total_df, x, on="Item_ID")
    total_df.set_index("User_ID", inplace=True)
    total_df = total_df.reset_index().sort_values(by="User_ID")
    total_df.head(100).to_csv("hi.csv", encoding="utf-8", escapechar="\\")
    """
    User_Topics = user_id_topics_avg(x, total_df)
    User_Topics.to_csv("user_topics.csv", encoding="utf-8", escapechar="\\")
