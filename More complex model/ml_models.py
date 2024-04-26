# Collection of ML models used for nuisance estimation
import numpy as np
import tensorflow as tf
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC


# Outcome regression function
def get_model_g(model_type, model_params): #consider setting seeds random_state
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror')
    elif model_type == 'rf':
        model = RandomForestRegressor()        
    elif model_type == 'svm':
        model = SVR()
    elif model_type == 'linear_svm':
        model = LinearSVR()
    elif model_type == 'lasso':
        model = LassoCV()
    elif model_type == 'ridge':
        model = RidgeCV()
    else model_type == 'nn':
        np.random.seed(42)
        tf.random.set_seed(42)
        model = Sequential()
        model.add(Input(shape=(10,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

    if model_params is not None:
        model.set_params(**model_params)

    return model

# Propensity score
def get_model_m(model_type, model_params): #consider setting seeds random_state
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(objective='binary:logistic')
    elif model_type == 'rf':
        model = RandomForestClassifier()        
    elif model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'linear_svm':
        model = LinearSVC()
    elif model_type == 'lasso':
        model = 
    elif model_type == 'ridge':
        model = 
    else model_type == 'nn':
        np.random.seed(42)
        tf.random.set_seed(42)
        model = Sequential()
        model.add(Input(shape=(10,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

    if model_params is not None:
        model.set_params(**model_params)

    return model
