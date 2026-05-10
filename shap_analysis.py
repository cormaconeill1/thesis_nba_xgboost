"""
shap_analysis.py

generates SHAP summary plots for each position model and each outcome class.
plots are saved to the figures directory for use in thesis reporting.
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

os.makedirs('../figures', exist_ok=True)

positions = ['guards', 'forwards', 'bigs']

for position in positions:
    df = pd.read_csv(f'../data/{position}.csv')
    model = joblib.load(f'../models/{position}_model.pkl')

    X = df.drop(columns=['Name', 'Year', 'Position', 'Label'])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    class_names = ['Bust', 'Role Player', 'Starter', 'Superstar']

    #one summary plot per class shows feature influence for that outcome
    for i, class_name in enumerate(class_names):
        shap.summary_plot(shap_values[:, :, i], X, show=False)
        plt.title(f'SHAP Summary - {position.capitalize()} - {class_name}')
        plt.tight_layout()
        plt.savefig(f'../figures/{position}_shap_{class_name.lower().replace(" ", "_")}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()