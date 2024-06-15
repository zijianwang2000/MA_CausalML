import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from data_generation import m_0, g_0, get_data

# Load tuned hyperparameters of XGBoost
with open('opt_params_xgboost.pkl', 'rb') as pickle_file:
    opt_params_xgboost = pickle.load(pickle_file)

# Get XGBoost models from hyperparameters
def get_models(xgb_params_dict):
    model_g = []
    for d in [0, 1]:
        model = xgb.XGBRegressor(objective='reg:squarederror', seed=0)
        model.set_params(**xgb_params_dict[f'g{d}'])
        model_g.append(model)
    model_m = xgb.XGBClassifier(objective='binary:logistic', seed=0)
    model_m.set_params(**xgb_params_dict['m'])
    return model_g, model_m

mse = {}

for N in opt_params_xgboost.keys():
    rng = np.random.default_rng(seed=23)
    mse_N = {}
    
    for j in range(3):
        y_data, d_data, x_data = get_data(N, rng)

        model_g, model_m = get_models(opt_params_xgboost[N][j])
        mse_N_j = {'g0':[], 'g1':[], 'm':[]}

        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for (train_indices, eval_indices) in skf.split(X=x_data, y=d_data):
            y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
            y_eval, d_eval, x_eval = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices]

            # Estimate outcome regression functions g_0(d)
            for d in [0, 1]:
                model_g[d].fit(X=x_train[d_train==d], y=y_train[d_train==d])
                mse_N_j[f'g{d}'].append(mean_squared_error(g_0(d, x_eval), model_g[d].predict(x_eval)))
                
            # Estimate propensity score m_0
            model_m.fit(X=x_train, y=d_train)
            mse_N_j['m'].append((mean_squared_error(m_0(x_eval), model_m.predict_proba(x_eval)[:,1]), mean_absolute_error(m_0(x_eval), model_m.predict_proba(x_eval)[:,1])))

        mse_N[j] = mse_N_j

    mse[N] = mse_N

with open('mse.pkl', 'wb') as pickle_file:
    pickle.dump(mse, pickle_file)

for N in opt_params_xgboost.keys():
    for j in range(3):
        for name in ['g0', 'g1', 'm']:
            print(N, j, name, np.round(np.mean(mse[N][j][name], axis=0), 6))