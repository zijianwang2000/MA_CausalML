# Monte Carlo simulation
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_generation import get_data
from dml_algorithm import mm_att, dml_parallel_att


# Load tuned hyperparameters of MLP
with open('opt_params_mlp.pkl', 'rb') as pickle_file:
    opt_params_mlp = pickle.load(pickle_file)


# Get MLP models from hyperparameters
def get_models(mlp_params_dict):
    model_g0 = MLPRegressor(hidden_layer_sizes=(32,16), tol=0.0005, n_iter_no_change=5, random_state=42)
    model_g0.set_params(**mlp_params_dict['g0'])
    model_m = MLPClassifier(hidden_layer_sizes=(32,16), tol=0.0005, n_iter_no_change=5, random_state=42)
    model_m.set_params(**mlp_params_dict['m'])
    return model_g0, model_m


# MC simulation for a given sample size N
def mc_simulation(N, n_MC=2000):
    rng = np.random.default_rng(seed=123)
    att_estimates = np.empty((n_MC, 2))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))
    rmses = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        scaler = StandardScaler()
        x_data_stand = scaler.fit_transform(x_data)
        
        att_estimates[j, 0] = mm_att(y_data, d_data, x_data)
        model_g0, model_m = get_models(opt_params_mlp[N][j])
        att_estimates[j, 1], sigma_estimates[j], CIs[j], rmses[j] = dml_parallel_att(y_data, d_data, [x_data, x_data_stand], model_g0, model_m, m_bounds=(0.005, 0.995))

    return [att_estimates, sigma_estimates, CIs, rmses]


# MC simulation for all sample sizes
results_dict = {}

for N in opt_params_mlp.keys():
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('results_dict_mlp.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
