# Monte Carlo simulation
import numpy as np
import xgboost as xgb
import pickle
from data_generation import get_data
from dml_algorithm import mm_ate, dml_parallel_ate


# Load tuned hyperparameters of XGBoost
with open('opt_params_xgboost.pkl', 'rb') as pickle_file:
    opt_params_xgboost = pickle.load(pickle_file)


# Get XGBoost models from hyperparameters
def get_models(xgb_params_dict):
    model_g = []
    for d in [0, 1]:
        model = xgb.XGBRegressor(objective='reg:squarederror', seed=0)
        model.set_params(**xgb_params_dict[f'g{d}'])
        model_g.append(model)
    model_m = xgb.XGBClassifier(objective='binary:logistic', seed=0)
    model_m.set_params(**xgb_params_dict['m'])
    return model_g, model_m


# MC simulation for a given sample size N
def mc_simulation(N, n_MC=2000):
    rng = np.random.default_rng(seed=123)
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))
    rmses = np.empty((n_MC, 3))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data)
        model_g, model_m = get_models(opt_params_xgboost[N][j])
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j], rmses[j] = dml_parallel_ate(y_data, d_data, x_data, model_g, model_m)

    return [ate_estimates, sigma_estimates, CIs, rmses]


# MC simulation for all sample sizes
results_dict = {}

for N in opt_params_xgboost.keys():
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('results_dict_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
