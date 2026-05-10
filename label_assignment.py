"""
label_assignment.py

assigns a career outcome label to each player in merged_final.csv based on
career win shares and all-star selections. labels are applied hierarchically
— a player is assessed from the highest tier downward and assigned the first
tier they meet.
0 — Bust: WS < 5
1 — Role Player: WS >= 5
2 — Starter: WS >= 20
3 — Superstar: WS >= 30 & All-Star appearances >= 2
"""

import pandas as pd

df = pd.read_csv('../data/merged_final.csv')

def assign_label(row):
    if row['WS'] >= 30 and row['All Star'] >= 2:
        return 3  #superstar
    elif row['WS'] >= 20:
        return 2  #starter
    elif row['WS'] >= 5:
        return 1  #role player
    else:
        return 0  #bust


df['Label'] = df.apply(assign_label, axis=1)

df.to_csv('../data/merged_final.csv', index=False)
print(df['Label'].value_counts().sort_index())
