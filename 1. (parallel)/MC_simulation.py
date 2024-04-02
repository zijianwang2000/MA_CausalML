# Monte Carlo simulation
import numpy as np
import sklearn
import xgboost as xgb
import multiprocessing as mp
import pickle

from sklearn.model_selection import GridSearchCV
from data_generation import m_0, g_0, get_data
from dml_algorithm import mm_ate, dml_ate


# Prepare cross-validation to find optimal hyperparameters of XGBoost
def xgb_cv(y_data, d_data, x_data, cv=5):
    xgb_model_g = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model_m = xgb.XGBClassifier(objective='binary:logistic')

    param_grid = {
        'n_estimators': [5, 10, 25, 50, 75, 100, 150, 200],
        'max_depth': [2, 3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.2, 0.3],
        'reg_lambda': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
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
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
xgb_params_dict_dict = {}

for N in sample_sizes:
    np.random.seed(100)
    y_data, d_data, x_data = get_data(N)
    xgb_params_dict_dict[N] = xgb_cv(y_data, d_data, x_data)

with open('opt_params_xgboost.pkl', 'wb') as pickle_file:
    pickle.dump(xgb_params_dict_dict, pickle_file)


# MC simulation for a given sample size N
def mc_simulation(N):
    np.random.seed(100)
    n_MC = 5000
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))
    xgb_params_dict = xgb_params_dict_dict[N]

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        model_g0, model_g1 = xgb.XGBRegressor(objective='reg:squarederror'), xgb.XGBRegressor(objective='reg:squarederror')
        model_g0.set_params(**xgb_params_dict['g0'])
        model_g1.set_params(**xgb_params_dict['g1'])
        model_g = [model_g0, model_g1]
        model_m = xgb.XGBClassifier(objective='binary:logistic')
        model_m.set_params(**xgb_params_dict['m'])
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j] = dml_ate(y_data, d_data, x_data, model_g, model_m)

    results_dict[N] = [ate_estimates, sigma_estimates, CIs]
    print(f'MC simulation done for N={N}')


# MC simulation for all sample sizes using multiprocessing
if __name__ == '__main__':
    results_dict = {}

    with mp.Pool(7) as pool:
        pool.map(mc_simulation, sample_sizes)

with open('results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
