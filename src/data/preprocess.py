import numpy as np
import pandas as pd
import os
import LDA


def process_dataset():
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
        :-23
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

    # (3) Create a mapping so we don't have to deal with random hashes for User_ID and Item_ID
    user_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["User_ID"].unique())
    }
    total_df["User_ID"] = total_df["User_ID"].map(user_id_mapping)

    item_id_mapping = {
        user_id: idx for idx, user_id in enumerate(total_df["Item_ID"].unique())
    }
    total_df["Item_ID"] = total_df["Item_ID"].map(item_id_mapping)
    print(total_df)

    Item_Topics = LDA.sum_reviews_LDA(total_df)
    Item_Topics.to_csv("IT.csv", encoding="utf-8", escapechar="\\")

    User_Topics = LDA.user_id_topics_avg(Item_Topics, total_df)
    User_Topics.to_csv("UT.csv", encoding="utf-8", escapechar="\\")


if __name__ == "__main__":
    process_dataset()
