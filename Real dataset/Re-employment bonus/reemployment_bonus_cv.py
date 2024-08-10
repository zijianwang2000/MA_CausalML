import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC


def preprocess():
    def get_quarter(row):
        for col, val in enumerate(row, start=1):
            if val == 1:
                return col
    
    def get_age(row):
        if (row == [1, 0]).all():
            return 0
        elif (row == [0, 0]).all():
            return 1
        elif (row == [0, 1]).all():
            return 2
            
    df = pd.read_csv("https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat", sep='\s+')
    df['log_dur'] = np.log(df['inuidur1'])
    df = df.drop(['abdt', 'inuidur1', 'inuidur2'], axis=1)
    df['quarter'] = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].apply(get_quarter, axis=1)
    df['age'] = df[['agelt35', 'agegt54']].apply(get_age, axis=1)
    df['white'] = 1 - df[['black', 'hispanic', 'othrace']].sum(axis=1)
    df = df.drop(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54', 'othrace'], axis=1)
    df = df.loc[df['tg'].isin([0, 4, 6]), :].copy()
    df['treatment'] = df['tg'].isin([4, 6]).astype('float')
    df = df.drop('tg', axis=1)

    return df


def induce_selection_bias():
    df = preprocess()  
    
    discard_0 = (1-df.treatment) * (df.white==1)
    discard_1 = df.treatment * ((df.female==0).astype(float) + (df.dep==0) + (df.recall==0) + (df.age==0) + (df.lusd==1))
    discard_prob = 0.5*discard_0 + 0.1*discard_1
    
    np.random.seed(123)
    discarded = (np.random.uniform(size=len(df)) <= discard_prob)
    df_bias = df[~discarded]
    df_bias = df_bias.sample(frac=1)
    
    return df_bias['log_dur'].values, df_bias['treatment'].values, df_bias.drop(['treatment', 'log_dur'], axis=1).values


# Prepare data
y_data, d_data, x_data = induce_selection_bias()

scaler = StandardScaler()
x_data_stand = scaler.fit_transform(x_data)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_data_quad = poly_features.fit_transform(x_data)
scaler = StandardScaler()
x_data_quad_stand = scaler.fit_transform(x_data_quad)


# Methods for performing cross-validation
def eln_cv(y_data, d_data, x_data):
    eln_model_g = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_alphas=100, max_iter=10000, n_jobs=-1)
    eln_model_m = LogisticRegressionCV(Cs=50, l1_ratios=[0, .1, .3, .5, .7, .9, .95, .99, 1],
                                       penalty='elasticnet', solver='saga', max_iter=50000,
                                       random_state=42, scoring='neg_brier_score', n_jobs=-1)

    eln_params_dict = {}
    for d in [0, 1]:
        eln_model_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        eln_params_dict[f'g{d}'] = {'alpha': eln_model_g.alpha_, 'l1_ratio': eln_model_g.l1_ratio_}
    eln_model_m.fit(X=x_data, y=d_data)
    eln_params_dict['m'] = {'C': eln_model_m.C_[0], 'l1_ratio': eln_model_m.l1_ratio_[0]}

    return eln_params_dict


def mlp_cv(y_data, d_data, x_data, cv=5):
    mlp_model_g = MLPRegressor(random_state=42)
    mlp_model_m = MLPClassifier(random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(16,8,4), (32,16), (24,16), (16,8), (8,4)],
        'alpha': [0.01, 0.03, 0.1, 0.3, 1],
        'batch_size': [2, 4, 8, 16, 32],
        'max_iter': [25, 35, 50, 75, 100, 125]
    }

    grid_search_g = GridSearchCV(estimator=mlp_model_g, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=mlp_model_m, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='neg_brier_score')

    mlp_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        mlp_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    mlp_params_dict['m'] = grid_search_m.best_params_

    return mlp_params_dict


def rf_cv(y_data, d_data, x_data, cv=5):
    rf_model_g = RandomForestRegressor(random_state=42)
    rf_model_m = RandomForestClassifier(criterion='log_loss', random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300, 500, 750, 1000],
        'max_features': ['sqrt', 8, None],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_g = GridSearchCV(estimator=rf_model_g, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=rf_model_m, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_brier_score')

    rf_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        rf_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    rf_params_dict['m'] = grid_search_m.best_params_

    return rf_params_dict


def svm_cv(y_data, d_data, x_data, cv=5):
    svm_model_g = SVR()
    svm_model_m = SVC(probability=True, random_state=42)

    param_grid_m = {
        'kernel': ['linear', 'rbf'],
        'C': [0.01, 0.1, 0.3, 1, 3, 10, 100],
        'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3]
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


def xgb_cv(y_data, d_data, x_data, cv=5):
    xgb_model_g = xgb.XGBRegressor(objective='reg:squarederror', seed=0)
    xgb_model_m = xgb.XGBClassifier(objective='binary:logistic', seed=0)

    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [2, 3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'learning_rate': [0.02, 0.05, 0.1, 0.2, 0.3],
        'reg_lambda': [0.01, 0.1, 1, 10, 100],
        'reg_alpha': [0, 0.01, 0.1, 1, 10, 100]
    }

    grid_search_g = GridSearchCV(estimator=xgb_model_g, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_m = GridSearchCV(estimator=xgb_model_m, param_grid=param_grid, cv=cv, n_jobs=-1,
                                 scoring='neg_brier_score')

    xgb_params_dict = {}
    for d in [0, 1]:
        grid_search_g.fit(X=x_data[d_data == d], y=y_data[d_data == d])
        xgb_params_dict[f'g{d}'] = grid_search_g.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    xgb_params_dict['m'] = grid_search_m.best_params_

    return xgb_params_dict


# Perform cross-validation
eln_params_dict = eln_cv(y_data, d_data, x_data_quad_stand)
mlp_params_dict = mlp_cv(y_data, d_data, x_data_stand)
rf_params_dict = rf_cv(y_data, d_data, x_data)
svm_params_dict = svm_cv(y_data, d_data, x_data_stand)
xgb_params_dict = xgb_cv(y_data, d_data, x_data)

opt_params = {
    'eln': eln_params_dict, 
    'mlp': mlp_params_dict, 
    'rf': rf_params_dict,
    'svm': svm_params_dict, 
    'xgb': xgb_params_dict
}

with open('opt_params.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params, pickle_file)
