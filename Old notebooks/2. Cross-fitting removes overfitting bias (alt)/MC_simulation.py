#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import xgboost as xgb
import json
import pickle

from sklearn.model_selection import GridSearchCV
from data_generation import m_0, g_0, get_data
from dml_algorithm import dml_ate


# ## Load/Tune hyperparameters for XGBoost

# In[2]:


with open('opt_params_xgboost.json', 'r') as json_file:
    opt_params_dict_dict = json.load(json_file)


# In[3]:


N = 800

cf_settings = {
    'No cross-fitting': opt_params_dict_dict['1000'],
    r'$K=2$': opt_params_dict_dict['500'],
    r'$K=5$': None
}


# In[4]:


xgb_model_g = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model_m = xgb.XGBClassifier(objective='binary:logistic')

param_grid = {
    'n_estimators': [5, 10, 25, 50, 75, 100, 150, 200],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.2, 0.3],
    'reg_lambda': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
}

grid_search_g = GridSearchCV(estimator=xgb_model_g, param_grid=param_grid, cv=5, n_jobs=-1,
                             scoring='neg_mean_squared_error')
grid_search_m = GridSearchCV(estimator=xgb_model_m, param_grid=param_grid, cv=5, n_jobs=-1, 
                             scoring='neg_brier_score')


# In[ ]:


np.random.seed(123)
    
y_data, d_data, x_data = get_data(N)
opt_params_dict = {}
    
for d in [0, 1]:
    grid_search_g.fit(X=x_data[d_data==d], y=y_data[d_data==d])
    opt_params_dict[f'g{d}'] = grid_search_g.best_params_
   
grid_search_m.fit(X=x_data, y=d_data)
opt_params_dict['m'] = grid_search_m.best_params_
    
cf_settings[r'$K=5$'] = opt_params_dict


# In[ ]:


with open('cf_settings.pkl', 'wb') as pickle_file:
    pickle.dump(cf_settings, pickle_file)


# ## Infeasible method-of-moments estimator

# In[ ]:


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

# In[ ]:


np.random.seed(100)
n_MC = 5000
ate_estimates = np.empty((n_MC, 4))
sigma_estimates = np.empty((n_MC, 2))
CIs = np.empty((n_MC, 4))

for j in range(n_MC):
    y_data, d_data, x_data = get_data(N)
    ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data) 
    
    for setting, opt_params_dict in cf_settings.items():
        model_g0, model_g1 = xgb.XGBRegressor(objective='reg:squarederror'), xgb.XGBRegressor(objective='reg:squarederror')
        model_g0.set_params(**opt_params_dict['g0'])
        model_g1.set_params(**opt_params_dict['g1'])
        model_g = [model_g0, model_g1]
        model_m = xgb.XGBClassifier(objective='binary:logistic')
        model_m.set_params(**opt_params_dict['m'])
        
        if setting == 'No cross-fitting':
            ate_estimates[j, 1] = dml_no_cf_ate(y_data, d_data, x_data, model_g, model_m)
        elif setting == r'$K=2$':
            ate_estimates[j, 2], sigma_estimates[j, 0], CIs[j, :2] = dml_ate(2, y_data, d_data, x_data, model_g, model_m, classical=False)
        elif setting == r'$K=5$':
            ate_estimates[j, 3], sigma_estimates[j, 1], CIs[j, 2:] = dml_ate(5, y_data, d_data, x_data, model_g, model_m, classical=False)


# In[ ]:


np.save('ate_estimates.npy', ate_estimates)
np.save('sigma_estimates.npy', sigma_estimates)
np.save('CIs.npy', CIs)

