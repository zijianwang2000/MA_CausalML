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


# ## Infeasible method-of-moments estimator

# In[3]:


def mm_ate(y_data, d_data, x_data):
    return np.mean(g_0(1, x_data) - g_0(0, x_data) + d_data*(y_data-g_0(1, x_data))/m_0(x_data)
                   - (1-d_data)*(y_data-g_0(0, x_data))/(1-m_0(x_data)))


# ## MC simulation

# In[4]:


np.random.seed(100)
n_MC = 5000
K = 5
results_dict = {}

for N, opt_params_dict in opt_params_dict_dict.items():
    ate_estimates = np.empty((n_MC, 4))
    sigma_estimates = np.empty(n_MC)
    CIs = np.empty((n_MC, 2))
    for j in range(n_MC):
        y_data, d_data, x_data = get_data(int(N))
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data)
        model_g0, model_g1 = xgb.XGBRegressor(objective='reg:squarederror'), xgb.XGBRegressor(objective='reg:squarederror')
        model_g0.set_params(**opt_params_dict['g0'])
        model_g1.set_params(**opt_params_dict['g1'])
        model_g = [model_g0, model_g1]
        model_m = xgb.XGBClassifier(objective='binary:logistic')
        model_m.set_params(**opt_params_dict['m'])
        ate_estimates[j, 1:], sigma_estimates[j], CIs[j] = dml_ate(K, y_data, d_data, x_data, model_g, model_m)
    results_dict[int(N)] = [ate_estimates, sigma_estimates, CIs]

with open('results_dict.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)

