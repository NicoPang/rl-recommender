import numpy as np
import pandas as pd
import os
import LDA

subsets = { 'Toys_and_Games_5.json', 'reviews_Toys_and_Games.json',
           'Apps_for_Android_5.json','reviews_Apps_for_Android.json',
           'Health_and_Personal_Care_5.json','reviews_Health_and_Personal_Care.json'
}
total_df = pd.DataFrame(columns=['User_ID', 'Item_ID', 'Review_Body', 'Review_Score', 'Review_Header'])
total_df.set_index('User_ID', inplace=True)

# (1) Check if datasets on system
git_path = os.path.abspath(__file__)[:-23] #make sure not to change current filename or folder names
print(git_path)
for name in subsets:
    if '5' not in name: continue
    p = os.path.join(git_path, 'datasets', 'raw', name)
    if os.path.exists(path=p) is False:
        print(f"ERROR: issue not finding unzipped file at {p}")
    
    # (2) Combine the unzipped json datasets
    df = pd.read_json(path_or_buf=p, lines=True)

    df = df.drop(['reviewerName','helpful', 'unixReviewTime', 'reviewTime'], axis=1)
    df = df.rename(columns={'reviewerID':'User_ID',
                            'asin': 'Item_ID',
                            'reviewText': 'Review_Body',
                            'summary': 'Review_Header',
                            'overall': 'Review_Score'
                            })
    total_df = pd.concat([df, total_df], axis=0)

total_df = total_df.reset_index().sort_values(by='User_ID')
total_df.to_csv('hi.csv', encoding='utf-8', escapechar='\\')
