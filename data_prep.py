"""
data_prep.py

merges raw draft player records with career outcomes for the 2010-2020 draft
classes. produces merged_2010_2020.csv as input to data_cleaning.py.

name matching requires layered normalisation because the same player appears
under different forms across sources (unicode accents, college disambiguation
tags, initial dots, suffixes, nicknames).
"""

import pandas as pd
import unicodedata
import re

suffixes = {'jr', 'sr', 'ii', 'iii', 'iv'}

#manual overrides for cases rule based normalisation cannot resolve (legal name changes, nicknames etc)
manual_mapping = {
    'enes kanter': 'enes freedom',
    'chu maduabum': 'chukwudiebere maduabum',
    'didi louzada silva': 'didi louzada',
    'sviatoslav mykhailiuk': 'svi mykhailiuk',
    'wesley iwundu': 'wes iwundu',
    'juan hernangomez': 'juancho hernangomez',
    'mohamed bamba': 'mo bamba',
    'nicolas claxton': 'nic claxton',
    'iggy brazdeikis': 'ignas brazdeikis',
    'jeffery taylor': 'jeff taylor',
    'kenyon martin': 'kj martin',
    'walter tavares': 'edy tavares',
}

def normalize_name(name):
    if not isinstance(name, str):
        return name
    #special characters that survive ascii encoding
    name = name.replace('ß', 'ss').replace('İ', 'I').replace('ı', 'i')
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower().strip()
    name = re.sub(r'\(.*?\)', '', name).strip()
    name = re.sub(r'(?<=\b\w)\.(?=\s|\b)', '', name).strip()
    tokens = name.split()
    if tokens and tokens[-1].rstrip('.') in suffixes:
        tokens = tokens[:-1]
    name = ' '.join(tokens)
    return manual_mapping.get(name, name)

draft_players_full = pd.read_csv('../data/draft_players.csv')
outcomes_full = pd.read_csv('../data/outcomes.csv')

draft_players = draft_players_full[(draft_players_full['Year'] >= 2010) & (draft_players_full['Year'] <= 2020)].copy()
outcomes = outcomes_full[(outcomes_full['Year'] >= 2010) & (outcomes_full['Year'] <= 2020)].copy()

print(f"Draft players (2010-2020): {len(draft_players)}")
print(f"Outcomes (2010-2020): {len(outcomes)}")

draft_players['name_key'] = draft_players['Name'].apply(normalize_name)
outcomes['name_key'] = outcomes['Player'].apply(normalize_name)

#inner merge makes it retain every rows' pre-draft features and outcomes
merged_data = pd.merge(draft_players, outcomes, left_on=['name_key', 'Year'], right_on=['name_key', 'Year'], how='inner')
merged_data = merged_data.drop(columns=['name_key'])

print(f"Merged dataset rows: {len(merged_data)}")
print(f"Merged dataset columns: {len(merged_data.columns)}")

merged_data.to_csv('../data/merged_2010_2020.csv', index=False)
print(f"Merged dataset saved as 'merged_2010_2020.csv' with {len(merged_data)} records.")
