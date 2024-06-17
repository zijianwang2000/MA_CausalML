# Cross-validation to find optimal hyperparameters of MLP
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from data_generation import get_data


def get_batch_sizes(N):
    if N <= 500:
        batch_sizes_g, batch_sizes_m = [2, 4], [4, 8, 16]
    elif N == 1000:
        batch_sizes_g, batch_sizes_m = [2, 4, 8], [4, 8, 16]
    elif N == 2000:
        batch_sizes_g, batch_sizes_m = [2, 4, 8, 16], [4, 8, 16, 32]
    elif N == 4000:
        batch_sizes_g, batch_sizes_m = [4, 8, 16, 32], [16, 32]
    else:
        batch_sizes_g, batch_sizes_m = [8, 16, 32], [16, 32]
    return batch_sizes_g, batch_sizes_m


def get_max_iters_m(N):
    if N >= 8000:
        return [25, 35]
    elif N == 4000:
        return [25, 35, 50]
    else:
        return [25, 35, 50, 75]


def mlp_cv(y_data, d_data, x_data, batch_sizes_g, batch_sizes_m, max_iters_m, cv=5):
    mlp_model_g = MLPRegressor(hidden_layer_sizes=(32,16), tol=0.0005, n_iter_no_change=5, random_state=42)
    mlp_model_m = MLPClassifier(hidden_layer_sizes=(32,16), tol=0.0005, n_iter_no_change=5, random_state=42)

    param_grid_g = {
        'alpha': [0.05, 0.1, 0.2, 0.3],
        'batch_size': batch_sizes_g,
        'max_iter': [50, 75, 100, 125]
    }
    param_grid_m = {
        'alpha': [0.05, 0.1, 0.2, 0.3],
        'batch_size': batch_sizes_m,
        'max_iter': max_iters_m
    }

    grid_search_g = GridSearchCV(estimator=mlp_model_g, param_grid=param_grid_g, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=mlp_model_m, param_grid=param_grid_m, cv=cv, n_jobs=-1, scoring='neg_brier_score')

    mlp_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        mlp_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    mlp_params_dict['m'] = grid_search_m.best_params_

    return mlp_params_dict


# Perform cross-validation for all data sets
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
n_MC = 2000
opt_params_mlp = {}

for N in sample_sizes:
    rng = np.random.default_rng(seed=123)
    opt_params_mlp_N = {}
    batch_sizes_g, batch_sizes_m = get_batch_sizes(N)
    max_iters_m = get_max_iters_m(N)
    
    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        scaler = StandardScaler()
        x_data_stand = scaler.fit_transform(x_data)
        opt_params_mlp_N[j] = mlp_cv(y_data, d_data, x_data_stand, batch_sizes_g, batch_sizes_m, max_iters_m)

    opt_params_mlp[N] = opt_params_mlp_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_mlp.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_mlp, pickle_file)
