from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def fine_tune_random_forest(X, y, cv=4, n_jobs=-1, random_state=42):

    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_macro", verbose=1, n_jobs=n_jobs
    )
    grid.fit(X, y)
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_
