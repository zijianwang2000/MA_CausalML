# Cross-validation to find optimal hyperparameters of Random Forest
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_generation import get_data


def rf_cv(y_data, d_data, x_data, cv=5):
    rf_model_g = RandomForestRegressor(random_state=42)
    rf_model_m = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 250, 500, 1000, 2000],
        'max_features': ['sqrt', None],
        'max_depth': [2, 5, 10, 25, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search_g = GridSearchCV(estimator=rf_model_g, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=rf_model_m, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_brier_score')

    rf_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        rf_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    rf_params_dict['m'] = grid_search_m.best_params_

    return rf_params_dict


# Perform cross-validation for all data sets
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
n_MC = 2000
opt_params_rf = {}

for N in sample_sizes:
    rng = np.random.default_rng(seed=123)
    opt_params_rf_N = {}
    
    for j in range(n_MC): 
        y_data, d_data, x_data = get_data(N, rng)
        opt_params_rf_N[j] = rf_cv(y_data, d_data, x_data)

    opt_params_rf[N] = opt_params_rf_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_rf.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_rf, pickle_file)
