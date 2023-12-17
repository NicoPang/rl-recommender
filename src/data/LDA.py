from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import spacy
import os, gc

"""
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
"""
nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("es_core_news_sm")


def create_topics(pth: str, df: pd.DataFrame, s: str, n: int):
    if (s != "User_ID") and (s != "Item_ID"):
        raise Exception(f"{s} is not Item_ID or User_ID")
    if n <= 0:
        raise Exception("n needs to be greater then zero and be an int")

    grouped_review_df = (
        df.groupby(s)["Review_Body"].apply(lambda x: " ".join(x)).reset_index()
    )
    grouped_review_df = grouped_review_df.reset_index(drop=True)
    grouped_review_df = grouped_review_df.sort_values(by=s)

    """
    def process_text(text):
        doc = nlp(text)
        processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        return processed_text

    grouped_review_df["Review_Body"] = grouped_review_df["Review_Body"].apply(
        process_text
    )
    """

    s_list = grouped_review_df["Review_Body"].tolist()

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#examples-using-sklearn-feature-extraction-text-countvectorizer
    vectorizer = TfidfVectorizer(encoding="utf-8", stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(s_list)
    topics_n = n
    top_n_words = 30

    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=topics_n, random_state=42, n_jobs=-1)
    lda.fit(X)
    document_topics = lda.transform(X)
    tf_feature_names = vectorizer.get_feature_names_out()

    apple = os.path.join(pth, "datasets", "processed", "lda", f"{s}_topics.txt")
    with open(
        os.path.join(pth, "datasets", "processed", "lda", f"{s}_topics.txt"), "w"
    ) as f_file:
        all_text = str()
        for i, topic in enumerate(lda.components_):
            l = f"Topic_{i}: "
            top_word_indices = topic.argsort()[-top_n_words:][::-1]
            for j in top_word_indices:
                word = tf_feature_names[j]
                l += word + ", "
            l += "\n"
            all_text += l
        # print(all_text)
        f_file.write(all_text)

    df = pd.DataFrame(document_topics)
    df.index.name = s

    return df


def preprocess_text(dataset: set, pth: str, n_topics: int):
    if len(dataset) == 0:
        raise Exception("dataset is empty")

    # (A) Setting up main dataframe
    total_df = pd.DataFrame(
        columns=["User_ID", "Item_ID", "Review_Body", "Review_Score", "Review_Header"]
    )

    # (B) Adding info to main dataframe from subsets
    for name in dataset:
        p = os.path.join(pth, "datasets", "raw", name)
        if os.path.exists(path=p) is False:
            Exception(f"ERROR: issue not finding unzipped file at {p}")
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

    # (C) save total_df
    total_df["Review_Score"] = total_df["Review_Score"].apply(
        lambda x: str(x).replace("\\", "")
    )
    total_df["Review_Body"] = total_df["Review_Body"].apply(
        lambda x: str(x).replace("\\", "")
    )
    total_df.to_csv(
        os.path.join(pth, "datasets", "processed", "lda", "total_df.csv.gz"),
        escapechar="\\",
    )

    # (D) rename User_ID and Item_ID to index
    user_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["User_ID"].unique())
    }
    pd.DataFrame(data=user_id_mapping.keys()).to_csv(
        os.path.join(pth, "datasets", "processed", "lda", "user_mappings.csv.gz"),
    )
    total_df["User_ID"] = total_df["User_ID"].map(user_id_mapping)
    del user_id_mapping

    item_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["Item_ID"].unique())
    }
    pd.DataFrame(data=item_id_mapping.keys()).to_csv(
        os.path.join(pth, "datasets", "processed", "lda", "item_mappings.csv.gz"),
    )
    total_df["Item_ID"] = total_df["Item_ID"].map(item_id_mapping)
    del item_id_mapping
    gc.collect()

    # (E) Calculate topic distribution for each user and each item
    Item_Topic = create_topics(pth, total_df, s="Item_ID", n=n_topics)
    Item_Topic.to_csv(
        os.path.join(pth, "datasets", "processed", "lda", "item_topics.csv")
    )

    User_Topics = create_topics(pth, total_df, s="User_ID", n=n_topics)
    User_Topics.to_csv(
        os.path.join(pth, "datasets", "processed", "lda", "user_topics.csv")
    )


if __name__ == "__main__":
    subsets = {
        "Toys_and_Games_5.json",
        "Apps_for_Android_5.json",
        "Health_and_Personal_Care_5.json",
    }
    pth = os.path.abspath(__file__)[:-16]

    preprocess_text(dataset=subsets, pth=pth, n_topics=10)
