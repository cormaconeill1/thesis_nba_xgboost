"""
feature_selection.py

removes post-draft columns from the three position csv files, making it that
only pre-draft features are available to the models.
"""

import pandas as pd

#these columns reflect post-draft career performance, must be excluded
#prevent data leakage
post_draft_cols = [
    'Player', 'G_y', 'PPG', 'APG', 'Start%', 'All-NBA', 'All Star', 'All-Defense', '3P%_y', '3PM_36', 'PTS_36', 'AST_36', 'REB_36', 'OBPM_y', 'DBPM_y', 'TS%_y', 'USG%_y', 'TOV%', 'WS'
]

for filename in ['guards', 'forwards', 'bigs']:
    df = pd.read_csv(f'../data/{filename}.csv')
    df = df.drop(columns=post_draft_cols)
    df.to_csv(f'../data/{filename}.csv', index=False)

    print(f"{filename}: {len(df)} rows, {len(df.columns)} columns")