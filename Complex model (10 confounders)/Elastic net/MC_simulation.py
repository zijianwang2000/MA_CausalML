# Monte Carlo simulation
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from data_generation import get_data
from dml_algorithm import mm_ate, dml_ate


# Load tuned hyperparameters of ElasticNet
with open('opt_params_eln.pkl', 'rb') as pickle_file:
    opt_params_eln = pickle.load(pickle_file)


# Get ElasticNet models from hyperparameters
def get_models(eln_params_dict):
    model_g = []
    for d in [0, 1]:
        model = ElasticNet(max_iter=10000)
        model.set_params(**eln_params_dict[f'g{d}'])
        model_g.append(model)
    model_m = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=50000, random_state=42)
    model_m.set_params(**eln_params_dict['m'])
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
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        x_data_quad = poly_features.fit_transform(x_data)
        scaler = StandardScaler()
        x_data_quad_stand = scaler.fit_transform(x_data_quad)

        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data)
        model_g, model_m = get_models(opt_params_eln[N][j])
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j], rmses[j] = dml_ate(y_data, d_data, [x_data, x_data_quad_stand], model_g, model_m)

    return [ate_estimates, sigma_estimates, CIs, rmses]


# MC simulation for all sample sizes
results_dict = {}

for N in opt_params_eln.keys():
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
