# Cross-validation to find optimal hyperparameters of NN
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from data_generation import get_data


def mlp_cv(y_data, d_data, x_data, cv=5):
    mlp_model_g = MLPRegressor(random_state=42)
    mlp_model_m = MLPClassifier(random_state=42)

    param_grid_g = {
        'hidden_layer_sizes': [(8,4), (16,4), (16,8), (16,16), (32,16), (16,8,8), (16,16,8), (32,16,8), (32,32,16)],
        'alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 1],
        'batch_size': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'max_iter': [5, 10, 20, 30, 50, 75, 100, 150, 200],
        'tol': [1e-4, 0.001],
        'n_iter_no_change': [2, 5, 10]
    }
    param_grid_m = param_grid_g.copy()

    grid_search_g = RandomizedSearchCV(estimator=mlp_model_g, param_distributions=param_grid_g, n_iter=100, cv=cv, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error')
    grid_search_m = RandomizedSearchCV(estimator=mlp_model_m, param_distributions=param_grid_m, n_iter=100, cv=cv, n_jobs=-1, random_state=42, scoring='neg_brier_score')

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
    
    for j in range(5): 
        y_data, d_data, x_data = get_data(N, rng)
        opt_params_mlp_N[j] = mlp_cv(y_data, d_data, x_data)

    opt_params_mlp[N] = opt_params_mlp_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_mlp.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_mlp, pickle_file)

for N in sample_sizes:
    for j in range(5):
        print(opt_params_mlp[N][j])