"""
position_split.py

splits merged_final.csv into three position groups - guards, forwards, and
bigs, based on the position string recorded at draft time. produces three
csv files used by feature_selection.py.
"""

import pandas as pd

df = pd.read_csv('../data/merged_final.csv')

guards = ['PG', 'SG', 'PG/SG', 'SG/PG', 'SG/SF', 'SF/SG']
forwards = ['SF', 'SF/PF', 'PF/SF']
bigs = ['PF', 'C', 'PF/C', 'C/PF']

df_guards = df[df['Position'].isin(guards)]
df_forwards = df[df['Position'].isin(forwards)]
df_bigs = df[df['Position'].isin(bigs)]

df_guards.to_csv('../data/guards.csv', index=False)
df_forwards.to_csv('../data/forwards.csv', index=False)
df_bigs.to_csv('../data/bigs.csv', index=False)

print(f"Guards: {len(df_guards)}")
print(f"Forwards: {len(df_forwards)}")
print(f"Bigs: {len(df_bigs)}")
print(f"Total: {len(df_guards) + len(df_forwards) + len(df_bigs)}")