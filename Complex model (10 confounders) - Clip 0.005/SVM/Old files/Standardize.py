import numpy as np
import pickle
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from data_generation import m_0, g_0, get_data

# Load tuned hyperparameters of SVM
with open('opt_params_svm.pkl', 'rb') as pickle_file:
    opt_params_svm = pickle.load(pickle_file)
with open('opt_params_svm_stand.pkl', 'rb') as pickle_file:
    opt_params_svm_stand = pickle.load(pickle_file)

# Get SVM models from hyperparameters
def get_models(svm_params_dict):
    model_g = []
    for d in [0, 1]:
        model = SVR()
        model.set_params(**svm_params_dict[f'g{d}'])
        model_g.append(model)
    model_m = SVC(probability=True, random_state=42)
    model_m.set_params(**svm_params_dict['m'])
    return model_g, model_m

mse, mse_stand = {}, {}

for N in opt_params_svm.keys():
    rng = np.random.default_rng(seed=23)
    mse_N, mse_stand_N = {}, {}
    
    for j in range(3):
        y_data, d_data, x_data = get_data(N, rng)
        scaler = StandardScaler()
        x_data_stand = scaler.fit_transform(x_data)

        model_g, model_m = get_models(opt_params_svm[N][j])
        model_g_stand, model_m_stand = get_models(opt_params_svm_stand[N][j])
        mse_N_j, mse_stand_N_j = {'g0':[], 'g1':[], 'm':[]}, {'g0':[], 'g1':[], 'm':[]}

        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for (train_indices, eval_indices) in skf.split(X=x_data, y=d_data):
            y_train, d_train, x_train, x_train_stand = y_data[train_indices], d_data[train_indices], x_data[train_indices], x_data_stand[train_indices]
            y_eval, d_eval, x_eval, x_eval_stand = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices], x_data_stand[eval_indices]

            # Estimate outcome regression functions g_0(d)
            for d in [0, 1]:
                model_g[d].fit(X=x_train[d_train==d], y=y_train[d_train==d])
                mse_N_j[f'g{d}'].append(mean_squared_error(g_0(d, x_eval), model_g[d].predict(x_eval)))
                model_g_stand[d].fit(X=x_train_stand[d_train==d], y=y_train[d_train==d])
                mse_stand_N_j[f'g{d}'].append(mean_squared_error(g_0(d, x_eval), model_g_stand[d].predict(x_eval_stand)))
                
            # Estimate propensity score m_0
            model_m.fit(X=x_train, y=d_train)
            mse_N_j['m'].append((mean_squared_error(m_0(x_eval), model_m.predict_proba(x_eval)[:,1]), mean_absolute_error(m_0(x_eval), model_m.predict_proba(x_eval)[:,1])))
            model_m_stand.fit(X=x_train_stand, y=d_train)
            mse_stand_N_j['m'].append((mean_squared_error(m_0(x_eval), model_m_stand.predict_proba(x_eval_stand)[:,1]), mean_absolute_error(m_0(x_eval), model_m_stand.predict_proba(x_eval_stand)[:,1])))

        mse_N[j] = mse_N_j
        mse_stand_N[j] = mse_stand_N_j

    mse[N] = mse_N
    mse_stand[N] = mse_stand_N

with open('mse.pkl', 'wb') as pickle_file:
    pickle.dump(mse, pickle_file)
with open('mse_stand.pkl', 'wb') as pickle_file:
    pickle.dump(mse_stand, pickle_file)

for N in opt_params_svm.keys():
    for j in range(3):
        for name in ['g0', 'g1', 'm']:
            print(N, j, name, 
                  np.round(np.mean(mse[N][j][name], axis=0), 6), 
                  np.round(np.mean(mse_stand[N][j][name], axis=0), 6))