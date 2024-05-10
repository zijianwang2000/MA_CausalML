# Cross-validation to find optimal hyperparameters of ElasticNet
import numpy as np
import pickle
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from data_generation import get_data


def eln_cv(y_data, d_data, x_data):
    eln_model_g = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_alphas=100, max_iter=10000, n_jobs=-1)
    eln_model_m = LogisticRegressionCV(Cs=25, l1_ratios=[0, .1, .3, .5, .7, .9, .95, .99, 1],
                                       penalty='elasticnet', solver='saga', max_iter=50000,
                                       random_state=42, scoring='neg_brier_score', n_jobs=-1)

    eln_params_dict = {}
    for d in [0, 1]:
        eln_model_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        eln_params_dict[f'g{d}'] = {'alpha': eln_model_g.alpha_, 'l1_ratio': eln_model_g.l1_ratio_}
    eln_model_m.fit(X=x_data, y=d_data)
    eln_params_dict['m'] = {'C': eln_model_m.C_[0], 'l1_ratio': eln_model_m.l1_ratio_[0]}

    return eln_params_dict


# Perform cross-validation for all data sets
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
n_MC = 2000
opt_params_eln = {}

for N in sample_sizes:
    rng = np.random.default_rng(seed=123)
    opt_params_eln_N = {}
    
    for j in range(n_MC): 
        y_data, d_data, x_data = get_data(N, rng)
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        x_data_quad = poly_features.fit_transform(x_data)
        scaler = StandardScaler()
        x_data_quad_stand = scaler.fit_transform(x_data_quad)
        opt_params_eln_N[j] = eln_cv(y_data, d_data, x_data_quad_stand)

    opt_params_eln[N] = opt_params_eln_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_eln.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_eln, pickle_file)
