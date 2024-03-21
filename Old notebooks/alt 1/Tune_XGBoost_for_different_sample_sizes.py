#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import xgboost as xgb
import json

from sklearn.model_selection import GridSearchCV 
from data_generation import get_data


# ## Define list of sample sizes

# In[2]:


sample_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
opt_params_dict_dict = {}


# ## Use cross-validation to find optimal hyperparameters of XGBoost for each sample size

# In[3]:


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


# In[4]:


np.random.seed(123)

for N in sample_sizes:    
    y_data, d_data, x_data = get_data(N)
    opt_params_dict = {}
    
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data==d], y=y_data[d_data==d])
        opt_params_dict[f'g{d}'] = grid_search_g.best_params_
   
    grid_search_m.fit(X=x_data, y=d_data)
    opt_params_dict['m'] = grid_search_m.best_params_
    
    opt_params_dict_dict[N] = opt_params_dict
    print(f'Cross-validation done for N={N}')


# In[5]:


with open('opt_params_xgboost.json', 'w') as json_file:
    json.dump(opt_params_dict_dict, json_file)

