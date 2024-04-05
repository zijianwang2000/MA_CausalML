# DML algorithm
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm


# Infeasible method-of-moments estimator
def mm_ate(y_data, d_data, x_data, g_0, m_0):
    return np.mean(g_0(1, x_data) - g_0(0, x_data) + d_data*(y_data-g_0(1, x_data))/m_0(x_data)
                   - (1-d_data)*(y_data-g_0(0, x_data))/(1-m_0(x_data)))


# DML estimator without cross-fitting
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


# DML estimator with cross-fitting
def dml_ate(y_data, d_data, x_data, model_g, model_m, K=5, classical=True, inference=True, alpha=0.05):
    # Generate random partition of data for cross-fitting
    N = len(y_data)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # Compute respective ML estimators and thereupon auxiliary estimators
    theta_0_check_list = []
    if classical:
        reg_check_list, ipw_check_list = [], []
    if inference:
        scores_list = []
    
    for (train_indices, eval_indices) in skf.split(X=x_data, y=d_data):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_eval, d_eval, x_eval = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices]

        # Estimate outcome regression functions g_0(d)
        g_0_hat = []
        for d in [0, 1]:
            model_g[d].fit(X=x_train[d_train==d], y=y_train[d_train==d])
            g_0_hat.append(model_g[d].predict(x_eval))

        # Estimate propensity score m_0
        model_m.fit(X=x_train, y=d_train)
        m_0_hat = model_m.predict_proba(x_eval)[:,1]
            
        # Compute auxiliary estimator
        scores = g_0_hat[1] - g_0_hat[0] + d_eval*(y_eval-g_0_hat[1])/m_0_hat - (1-d_eval)*(y_eval-g_0_hat[0])/(1-m_0_hat)
        theta_0_check_list.append(np.mean(scores))

        # For variance estimation
        if inference:
            scores_list.append(scores)

        # For regression & IPW estimators
        if classical:
            reg_check_list.append(np.mean(g_0_hat[1] - g_0_hat[0])) 
            ipw_check_list.append(np.mean(d_eval*y_eval/m_0_hat - (1-d_eval)*y_eval/(1-m_0_hat)))     

    # Compute final estimator
    theta_0_hat = np.mean(theta_0_check_list)
    if classical:
        reg_hat, ipw_hat = np.mean(reg_check_list), np.mean(ipw_check_list)

    # Inference: estimate variance and construct confidence interval
    if inference:
        sigma_hat = np.sqrt(np.mean((np.array(scores_list)-theta_0_hat)**2))
        quantile = norm.ppf(1-alpha/2)
        CI = np.array([theta_0_hat-quantile*sigma_hat/np.sqrt(N), theta_0_hat+quantile*sigma_hat/np.sqrt(N)])

    # Return results
    if classical:
        if inference:
            return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI
        else:
            return np.array([theta_0_hat, reg_hat, ipw_hat])
    else:
        if inference:
            return theta_0_hat, sigma_hat, CI
        else:
            return theta_0_hat
