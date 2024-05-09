# Cross-validation to find optimal hyperparameters of SVM
import numpy as np
import pickle
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from data_generation import get_data


def svm_cv(y_data, d_data, x_data, cv=5):
    svm_model_g = SVR()
    svm_model_m = SVC(probability=True, random_state=42)

    param_grid_m = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 0.3, 1, 3, 10, 30, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    param_grid_g = param_grid_m.copy()
    param_grid_g['epsilon'] = [0.01, 0.03, 0.1, 0.3, 1]

    grid_search_g = GridSearchCV(estimator=svm_model_g, param_grid=param_grid_g, cv=cv, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=svm_model_m, param_grid=param_grid_m, cv=cv, n_jobs=-1,
                                 scoring='neg_brier_score')

    svm_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        svm_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    svm_params_dict['m'] = grid_search_m.best_params_

    return svm_params_dict


# Perform cross-validation for all data sets
sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
n_MC = 2000
opt_params_svm = {}

for N in sample_sizes:
    rng = np.random.default_rng(seed=123)
    opt_params_svm_N = {}
    
    for j in range(n_MC): 
        y_data, d_data, x_data = get_data(N, rng)
        scaler = StandardScaler()
        x_data_stand = scaler.fit_transform(x_data)
        opt_params_svm_N[j] = svm_cv(y_data, d_data, x_data_stand)

    opt_params_svm[N] = opt_params_svm_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_svm.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_svm, pickle_file)
