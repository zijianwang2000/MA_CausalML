#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import xgboost as xgb
import json
import pickle

from data_generation import m_0, g_0, get_data
from dml_algorithm import dml_ate


# ## Load tuned hyperparameters of XGBoost for each sample size

# In[2]:


with open('opt_params_xgboost.json', 'r') as json_file:
    opt_params_dict_dict = json.load(json_file)

with open('opt_params_xgboost_cf.pkl', 'rb') as pickle_file:
    opt_params_dict_dict_cf = pickle.load(pickle_file)


# ## Infeasible method-of-moments estimator

# In[3]:


def mm_ate(y_data, d_data, x_data):
    return np.mean(g_0(1, x_data) - g_0(0, x_data) + d_data*(y_data-g_0(1, x_data))/m_0(x_data)
                   - (1-d_data)*(y_data-g_0(0, x_data))/(1-m_0(x_data)))


# ## DML estimator without cross-fitting

# In[ ]:


def dml_no_cf_ate(y_data, d_data, x_data, model_g, model_m):
    # Estimate outcome regression functions g_0(d)
    g_0_hat = []
    for d in [0, 1]:
        model_g[d].fit(X=x_data[d_data==d], y=y_data[d_data==d])
        g_0_hat.append(model_g[d].predict(x_data))

    # Estimate propensity score m_0
    model_m.fit(X=x_data, y=d_data)
    m_0_hat = model_m.predict_proba(x_data)[:,1]

    return np.mean(g_0_hat[1] - g_0_hat[0] + d_data*(y_data-g_0_hat[1])/m_0_hat
                   - (1-d_data)*(y_data-g_0_hat[0])/(1-m_0_hat))


# ## MC simulation

# In[4]:


n_MC = 5000
results_dict = {}


for N, opt_params_dict in opt_params_dict_dict.items():
    np.random.seed(100)
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty((n_MC, 2))
    CIs = np.empty((n_MC, 4))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(int(N))
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data)

        # Loop over different settings of cross-fitting
        for l in range(3):
            # No cross-fitting
            if l == 0:
                opt_params_dict_ = opt_params_dict_dict_cf[int(N)//4*5]
            # K=2
            elif l == 1:
                opt_params_dict_ = opt_params_dict_dict_cf[int(N)//8*5]
            # K=5
            else:
                opt_params_dict_ = opt_params_dict

            model_g0, model_g1 = xgb.XGBRegressor(objective='reg:squarederror'), xgb.XGBRegressor(objective='reg:squarederror')
            model_g0.set_params(**opt_params_dict_['g0'])
            model_g1.set_params(**opt_params_dict_['g1'])
            model_g = [model_g0, model_g1]
            model_m = xgb.XGBClassifier(objective='binary:logistic')
            model_m.set_params(**opt_params_dict_['m'])

            # No cross-fitting
            if l == 0:
                ate_estimates[j, 1] = dml_no_cf_ate(y_data, d_data, x_data, model_g, model_m)
            # K=2
            elif l == 1:
                ate_estimates[j, 2], sigma_estimates[j, 0], CIs[j, :2] = dml_ate(2, y_data, d_data, x_data, model_g, model_m, classical=False)
            # K=5
            else:
                ate_estimates[j, 3], sigma_estimates[j, 1], CIs[j, 2:] = dml_ate(5, y_data, d_data, x_data, model_g, model_m, classical=False)

    results_dict[int(N)] = [ate_estimates, sigma_estimates, CIs]


with open('results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)

