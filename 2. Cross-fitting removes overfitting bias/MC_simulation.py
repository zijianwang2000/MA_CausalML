# Monte Carlo simulation
import numpy as np
import sklearn
import xgboost as xgb
import pickle

from data_generation import m_0, g_0, get_data
from dml_algorithm import mm_ate, dml_no_cf_ate, dml_ate


# Load tuned hyperparameters of XGBoost
with open('opt_params_xgboost.pkl', 'rb') as pickle_file:
    xgb_params_dict_dict = pickle.load(pickle_file)


# MC simulation for a given sample size N
def mc_simulation(N, model_g, model_m, n_MC=5000):
    np.random.seed(3)
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty((n_MC, 2))
    CIs = np.empty((n_MC, 4))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        ate_estimates[j, 1] = dml_no_cf_ate(y_data, d_data, x_data, model_g, model_m)
        ate_estimates[j, 2], sigma_estimates[j, 0], CIs[j, :2] = dml_ate(y_data, d_data, x_data, model_g, model_m, K=2, classical=False)
        ate_estimates[j, 3], sigma_estimates[j, 1], CIs[j, 2:] = dml_ate(y_data, d_data, x_data, model_g, model_m, K=5, classical=False)

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
