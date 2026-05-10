"""
model_training.py

trains a position-specific XGBoost multi-class classifier for each of the
three position groups. uses a chronological train/validation/test split to
reflect realistic draft evaluation conditions. saves each model to disk and
prints test set performance.
"""

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

positions = ['guards', 'forwards', 'bigs']

for position in positions:
    df = pd.read_csv(f'../data/{position}.csv')

    #chronological split prevents data leakage across draft classes
    train = df[df['Year'] <= 2016]
    val = df[(df['Year'] >= 2017) & (df['Year'] <= 2018)]
    test = df[df['Year'] >= 2019]

    X_train = train.drop(columns=['Name', 'Year', 'Position', 'Label'])
    y_train = train['Label']
    X_val = val.drop(columns=['Name', 'Year', 'Position', 'Label'])
    y_val = val['Label']
    X_test = test.drop(columns=['Name', 'Year', 'Position', 'Label'])
    y_test = test['Label']

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    base_model = XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', random_state=42)

    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1_macro', refit=True)
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    model = grid_search.best_estimator_

    joblib.dump(model, f'../models/{position}_model.pkl')

    y_pred = model.predict(X_test)

    print(f"\n {position} ")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Bust', 'Role Player', 'Starter', 'Superstar'], zero_division=0))