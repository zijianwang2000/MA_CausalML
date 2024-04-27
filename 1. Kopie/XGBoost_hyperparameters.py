# Cross-validation to find optimal hyperparameters of XGBoost
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import GridSearchCV
from data_generation import get_data


def xgb_cv(y_data, d_data, x_data, cv=5):
    xgb_model_g = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model_m = xgb.XGBClassifier(objective='binary:logistic')

    param_grid = {
        'n_estimators': [50, 75, 100, 150, 200],
        'max_depth': [2],
        'subsample': [0.6, 0.8],
        'learning_rate': [0.03, 0.06, 0.1],
        'reg_lambda': [0.01, 0.1, 1, 10, 100]
    }

    grid_search_g = GridSearchCV(estimator=xgb_model_g, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=xgb_model_m, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_brier_score')

    xgb_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        xgb_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    xgb_params_dict['m'] = grid_search_m.best_params_

    return xgb_params_dict


# Perform cross-validation for each sample size
np.random.seed(123)
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
xgb_params_dict_dict = {}

for N in sample_sizes:
    y_data, d_data, x_data = get_data(N)
    xgb_params_dict_dict[N] = xgb_cv(y_data, d_data, x_data)
    print(f'Cross-validation done for N={N}')

with open('opt_params_xgboost.pkl', 'wb') as pickle_file:
    pickle.dump(xgb_params_dict_dict, pickle_file)
