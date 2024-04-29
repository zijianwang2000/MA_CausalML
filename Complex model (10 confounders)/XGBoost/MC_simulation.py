# Monte Carlo simulation
import numpy as np
import sklearn
import xgboost as xgb
import pickle

from data_generation import m_0, g_0, get_data
from dml_algorithm import mm_ate, dml_ate


# Load tuned hyperparameters of XGBoost
with open('opt_params_xgboost.pkl', 'rb') as pickle_file:
    xgb_params_dict_dict = pickle.load(pickle_file)


# MC simulation for a given sample size N
def mc_simulation(N, model_g, model_m, n_MC=5000):
    #np.random.seed(3)
    rng = np.random.default_rng(seed=1)
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j] = dml_ate(y_data, d_data, x_data, model_g, model_m)

    return [ate_estimates, sigma_estimates, CIs]


# MC simulation for all sample sizes
results_dict = {}

for N, xgb_params_dict in xgb_params_dict_dict.items():
    model_g0, model_g1 = xgb.XGBRegressor(objective='reg:squarederror'), xgb.XGBRegressor(objective='reg:squarederror')
    model_g0.set_params(**xgb_params_dict['g0'])
    model_g1.set_params(**xgb_params_dict['g1'])
    model_g = [model_g0, model_g1]
    model_m = xgb.XGBClassifier(objective='binary:logistic')
    model_m.set_params(**xgb_params_dict['m'])
    results_dict[N] = mc_simulation(N, model_g, model_m)
    print(f'MC simulation done for N={N}')

with open('results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
