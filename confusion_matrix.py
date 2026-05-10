"""
confusion_matrix.py

generates confusion matrices for each position-specific model and the unified
model. plots are saved to the figures directory for use in thesis reporting.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

os.makedirs('../figures', exist_ok=True)

class_names = ['Bust', 'Role Player', 'Starter', 'Superstar']

models = {
    'guards': 'Guards',
    'forwards': 'Forwards',
    'bigs': 'Bigs',
    'unified': 'Unified'
}

for key, label in models.items():
    #unified model is evaluated on the combined test set across all positions
    if key == 'unified':
        dfs = [pd.read_csv(f'../data/{p}.csv') for p in ['guards', 'forwards', 'bigs']]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(f'../data/{key}.csv')

    test = df[df['Year'] >= 2019]
    X_test = test.drop(columns=['Name', 'Year', 'Position', 'Label'])
    y_test = test['Label']

    model = joblib.load(f'../models/{key}_model.pkl')
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {label}')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'../figures/cm_{key}.png', dpi=150, bbox_inches='tight')
    plt.close()