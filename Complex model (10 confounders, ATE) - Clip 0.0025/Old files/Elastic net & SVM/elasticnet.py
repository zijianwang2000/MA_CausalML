import numpy as np
import pickle
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
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
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        x_data_quad = poly_features.fit_transform(x_data)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        model_g0 = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], n_alphas=10, max_iter=5000, n_jobs=-1)
        model_g1 = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], n_alphas=10, max_iter=5000, n_jobs=-1)
        model_g = [model_g0, model_g1]
        model_m = LogisticRegressionCV(Cs=10, l1_ratios=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1],
                                       penalty='elasticnet', solver='saga', max_iter=1000,
                                       random_state=42, scoring='neg_brier_score', n_jobs=-1)
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j] = dml_ate(y_data, d_data, x_data, x_data_quad, model_g, model_m)

    return [ate_estimates, sigma_estimates, CIs]


# MC simulation for all sample sizes
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
results_dict = {}

for N in sample_sizes:
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('elasticnet_results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
