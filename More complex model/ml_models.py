# Collection of ML models used for nuisance estimation
import numpy as np
import tensorflow as tf
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC


# Outcome regression function
def get_model_g(model_type, model_params): #consider setting seeds random_state
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', seed=0)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=0)        
    elif model_type == 'svm':
        model = SVR()
    elif model_type == 'linear_svm':
        model = LinearSVR(dual=False)
    elif model_type == 'lasso':
        model = LassoCV()
    elif model_type == 'ridge':
        model = RidgeCV()
    elif model_type == 'nn':
        np.random.seed(42)
        tf.random.set_seed(42)
        model = Sequential()
        model.add(Input(shape=(10,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam())

    if model_params is not None:
        model.set_params(**model_params)

    return model

# Propensity score
def get_model_m(model_type, model_params): #consider setting seeds random_state
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(objective='binary:logistic', seed=0)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=0)        
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type == 'linear_svm':
        model = LinearSVC(dual=False)
    #elif model_type == 'lasso':
        #model = 
    #elif model_type == 'ridge':
        #model = 
    elif model_type == 'nn':
        np.random.seed(42)
        tf.random.set_seed(42)
        model = Sequential()
        model.add(Input(shape=(10,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam())

    if model_params is not None:
        model.set_params(**model_params)

    return model
