"""
process_ws.py

reads per-draft-class win share files from the WS directory and combines
them into a single win_shares.csv. output is used by data_cleaning.py
to append career win shares to the merged dataset.
"""

import pandas as pd

all_dfs = []

#each year's file follows the same Basketball-Reference draft table
for year in range(2010, 2021):
    filepath = f'../data/WS/{year}.txt'
    df = pd.read_csv(filepath, skiprows=3, header=1)
    df = df[['Pk', 'Player', 'WS']].copy()
    df['Year'] = year
    #drop header rows that Basketball-Reference repeats
    df = df[df['Player'].notna() & (df['Player'] != 'Player')]
    all_dfs.append(df)

win_shares = pd.concat(all_dfs, ignore_index=True)
win_shares['WS'] = pd.to_numeric(win_shares['WS'], errors='coerce').fillna(0.0)
win_shares['Pk'] = pd.to_numeric(win_shares['Pk'], errors='coerce')

win_shares.to_csv('../data/win_shares.csv', index=False)
print(f"Done. Total players: {len(win_shares)}")
