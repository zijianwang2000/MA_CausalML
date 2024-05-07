import numpy as np
import pickle
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from data_generation import m_0, g_0, get_data
from dml_algorithm import mm_ate, dml_ate


# MC simulation for a given sample size N
def mc_simulation(N, n_MC=5000):
    rng = np.random.default_rng(seed=123)
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        model_g0 = LinearSVR(C=5, dual='auto', max_iter=10000, random_state=42)
        model_g1 = LinearSVR(C=5, dual='auto', max_iter=10000, random_state=42)
        model_g = [model_g0, model_g1]
        model_m = CalibratedClassifierCV(estimator=LinearSVC(C=5, dual='auto', max_iter=10000, random_state=42))
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j] = dml_ate(y_data, d_data, x_data, model_g, model_m)

    return [ate_estimates, sigma_estimates, CIs]


# MC simulation for all sample sizes
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
results_dict = {}

for N in sample_sizes:
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('svm_results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
