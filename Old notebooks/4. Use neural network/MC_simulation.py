# Monte Carlo simulation
import numpy as np
import tensorflow as tf
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from data_generation import m_0, g_0, get_data
from dml_algorithm import mm_ate, dml_no_cf_ate, dml_ate


def get_model_g():
    np.random.seed(42)
    tf.random.set_seed(42)
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_model_m():
    np.random.seed(42)
    tf.random.set_seed(42)
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# MC simulation for a given sample size N
def mc_simulation(N, get_model_g, get_model_m, n_MC=5000):
    np.random.seed(1)
    y_data_list, d_data_list, x_data_list = [], [], []
    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N) 
        y_data_list.append(y_data)
        d_data_list.append(d_data)
        x_data_list.append(x_data)
       
    ate_estimates = np.empty((n_MC, 8))
    sigma_estimates = np.empty((n_MC, 2))
    CIs = np.empty((n_MC, 4))

    for j in range(n_MC):
        y_data, d_data, x_data = y_data_list[j], d_data_list[j], x_data_list[j]
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data, g_0, m_0)
        ate_estimates[j, 1] = dml_no_cf_ate(y_data, d_data, x_data, get_model_g, get_model_m)
        ate_estimates[j, 2:5], sigma_estimates[j, 0], CIs[j, :2] = dml_ate(y_data, d_data, x_data, get_model_g, get_model_m, K=2)
        ate_estimates[j, 5:], sigma_estimates[j, 1], CIs[j, 2:] = dml_ate(y_data, d_data, x_data, get_model_g, get_model_m, K=5)

    return [ate_estimates, sigma_estimates, CIs]


# MC simulation for all sample sizes
N = 2000
results_dict = mc_simulation(N, get_model_g, get_model_m)
print(f'MC simulation done for N={N}')

with open(f'results_dict_{N}.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)
