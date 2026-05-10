"""
data_cleaning.py

appends career win shares to the merged dataset and removes columns not
needed for modelling. produces merged_final.csv as input to label_assignment.py.
"""

import pandas as pd
import unicodedata

def normalize_name(name):
    if not isinstance(name, str):
        return name
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').strip()

df = pd.read_csv('../data/merged_2010_2020.csv')
win_shares = pd.read_csv('../data/win_shares.csv')

df['Name_normalized'] = df['Name'].apply(normalize_name)
win_shares['Player_normalized'] = win_shares['Player'].apply(normalize_name)

#left join retains all 660 players, unmatched ws filled in
df = pd.merge(df, win_shares[['Player_normalized', 'Year', 'WS']],
              left_on=['Name_normalized', 'Year'],
              right_on=['Player_normalized', 'Year'],
              how='left')

#drop columns not used as modelling features
df = df.drop(columns=['Name_normalized', 'Player_normalized', 'Birthdate', 'Draft Team', 'Team', 'Nation', 'Pk'])

df.to_csv('../data/merged_final.csv', index=False)
print(f"Final dataset saved as 'merged_final.csv' with {len(df)} records and {len(df.columns)} columns.")
