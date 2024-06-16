# DML algorithm
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.base import is_regressor
from scipy.stats import norm
from data_generation import g_0, m_0


# Infeasible method-of-moments estimator of the ATE
def mm_ate(y_data, d_data, x_data):
    return np.mean(g_0(1, x_data) - g_0(0, x_data) + d_data*(y_data-g_0(1, x_data))/m_0(x_data) - (1-d_data)*(y_data-g_0(0, x_data))/(1-m_0(x_data)))


# DML estimator of the ATE without cross-fitting
def dml_no_cf_ate(y_data, d_data, x_data, model_g, model_m, alpha=0.05, m_bounds=None):
    # Estimate outcome regression functions g_0(d)
    g_0_hat = []
    for d in [0, 1]:
        model_g[d].fit(X=x_data[d_data==d], y=y_data[d_data==d])
        g_0_hat.append(model_g[d].predict(x_data))

    # Estimate propensity score m_0
    model_m.fit(X=x_data, y=d_data)
    m_0_hat = model_m.predict_proba(x_data)[:,1]
    if m_bounds is not None:
        np.clip(m_0_hat, m_bounds[0], m_bounds[1], out=m_0_hat)
    
    # Compute ATE estimator
    scores = g_0_hat[1] - g_0_hat[0] + d_data*(y_data-g_0_hat[1])/m_0_hat - (1-d_data)*(y_data-g_0_hat[0])/(1-m_0_hat)
    estimate = np.mean(scores)

    # Inference: estimate standard deviation and construct confidence interval
    sigma_hat = np.sqrt(np.mean((scores-estimate)**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    CI = np.array([estimate-quantile*sigma_hat/np.sqrt(N), estimate+quantile*sigma_hat/np.sqrt(N)])

    return estimate, sigma_hat, CI


# DML estimator of the ATE with cross-fitting, without parallelization
def dml_ate(y_data, d_data, x_data_all, model_g, model_m, K=5, alpha=0.05, classical=True, errors=True, m_bounds=None):
    # Check for transformed input features
    if isinstance(x_data_all, list):
        x_data_orig, x_data = x_data_all[0], x_data_all[1]
    else:
        x_data_orig, x_data = x_data_all, x_data_all

    # Partition the data for cross-fitting
    skf = StratifiedKFold(n_splits=K, shuffle=False)

    # Compute respective ML estimators and thereupon auxiliary estimators
    theta_0_check_list = []
    scores_list = []
    if classical:
        reg_check_list, ipw_check_list = [], []
    if errors:
        rmse_list = []
    
    for (train_indices, eval_indices) in skf.split(X=x_data, y=d_data):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_eval, d_eval, x_eval, x_eval_orig = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices], x_data_orig[eval_indices]

        # Estimate outcome regression functions g_0(d)
        g_0_hat = []
        for d in [0, 1]:
            model_g[d].fit(X=x_train[d_train==d], y=y_train[d_train==d])
            g_0_hat.append(model_g[d].predict(x_eval))

        # Estimate propensity score m_0
        model_m.fit(X=x_train, y=d_train)
        m_0_hat = model_m.predict_proba(x_eval)[:,1]
        if m_bounds is not None:
            np.clip(m_0_hat, m_bounds[0], m_bounds[1], out=m_0_hat)
            
        # Compute auxiliary estimator
        scores = g_0_hat[1] - g_0_hat[0] + d_eval*(y_eval-g_0_hat[1])/m_0_hat - (1-d_eval)*(y_eval-g_0_hat[0])/(1-m_0_hat)
        theta_0_check_list.append(np.mean(scores))

        # For variance estimation
        scores_list.append(scores)

        # For regression & IPW estimators
        if classical:
            reg_check_list.append(np.mean(g_0_hat[1] - g_0_hat[0])) 
            ipw_check_list.append(np.mean(d_eval*y_eval/m_0_hat - (1-d_eval)*y_eval/(1-m_0_hat)))

        # Assess RMSE of ML models on evaluation set
        if errors:
            rmse_g0 = root_mean_squared_error(g_0(0, x_eval_orig), g_0_hat[0])
            rmse_g1 = root_mean_squared_error(g_0(1, x_eval_orig), g_0_hat[1])
            rmse_m = root_mean_squared_error(m_0(x_eval_orig), m_0_hat)
            rmse_list.append([rmse_g0, rmse_g1, rmse_m])

    # Compute final estimator
    theta_0_hat = np.mean(theta_0_check_list)
    if classical:
        reg_hat, ipw_hat = np.mean(reg_check_list), np.mean(ipw_check_list)

    # Inference: estimate standard deviation and construct confidence interval
    sigma_hat = np.sqrt(np.mean((np.array(scores_list)-theta_0_hat)**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    CI = np.array([theta_0_hat-quantile*sigma_hat/np.sqrt(N), theta_0_hat+quantile*sigma_hat/np.sqrt(N)])

    # Average RMSEs across folds
    if errors:
        rmse = np.mean(rmse_list, axis=0)

    # Return results
    if classical and errors:
        return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI, rmse
    elif classical and (not errors):
        return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI
    elif (not classical) and errors:
        return theta_0_hat, sigma_hat, CI, rmse
    else:
        return theta_0_hat, sigma_hat, CI


# DML estimator of the ATE with cross-fitting, parallelized
def dml_parallel_ate(y_data, d_data, x_data_all, model_g, model_m, K=5, alpha=0.05, classical=True, errors=True, m_bounds=None):
    # Check for transformed input features
    if isinstance(x_data_all, list):
        x_data_orig, x_data = x_data_all[0], x_data_all[1]
    else:
        x_data_orig, x_data = x_data_all, x_data_all

    # Process one data split in the cross-fitting procedure
    def process_single_split(train_indices, eval_indices):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_eval, d_eval, x_eval, x_eval_orig = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices], x_data_orig[eval_indices]

        # Estimate single nuisance function
        def fit_predict(model, X, y):
            model.fit(X, y)
            if is_regressor(model):
                return model.predict(x_eval)
            else:
                return model.predict_proba(x_eval)[:, 1]

        # Estimate outcome regression functions g_0(d) and propensity score m_0 in parallel
        model_data_list = [(model_g[d], x_train[d_train==d], y_train[d_train==d]) for d in [0, 1]] + [(model_m, x_train, d_train)]
        eta_0_hat = Parallel(n_jobs=3)(delayed(fit_predict)(model, X, y) for model, X, y in model_data_list)
        g_0_hat, m_0_hat = eta_0_hat[:2], eta_0_hat[2]
        if m_bounds is not None:
            np.clip(m_0_hat, m_bounds[0], m_bounds[1], out=m_0_hat)

        # Compute auxiliary estimator
        scores = g_0_hat[1] - g_0_hat[0] + d_eval*(y_eval-g_0_hat[1])/m_0_hat - (1-d_eval)*(y_eval-g_0_hat[0])/(1-m_0_hat)
        theta_0_check = np.mean(scores)

        # For regression & IPW estimators
        if classical:
            reg_check = np.mean(g_0_hat[1] - g_0_hat[0])
            ipw_check = np.mean(d_eval*y_eval/m_0_hat - (1-d_eval)*y_eval/(1-m_0_hat))
        else:
            reg_check, ipw_check = None, None

        # Assess RMSE of ML models on evaluation set
        if errors:
            rmse_g0 = root_mean_squared_error(g_0(0, x_eval_orig), g_0_hat[0])
            rmse_g1 = root_mean_squared_error(g_0(1, x_eval_orig), g_0_hat[1])
            rmse_m = root_mean_squared_error(m_0(x_eval_orig), m_0_hat)
        else:
            rmse_g0, rmse_g1, rmse_m = None, None, None

        return theta_0_check, reg_check, ipw_check, scores, [rmse_g0, rmse_g1, rmse_m]

    # Partition the data for cross-fitting
    skf = StratifiedKFold(n_splits=K, shuffle=False)

    # Cross-fitting, where the different splits are processed in parallel
    results = Parallel(n_jobs=K)(delayed(process_single_split)(train_indices, eval_indices) for train_indices, eval_indices in skf.split(X=x_data, y=d_data))

    # Collect results (in particular the auxiliary estimators)
    theta_0_check_list = [result[0] for result in results]
    scores_list = [result[3] for result in results]   # Needed for variance estimation
    if classical:
        reg_check_list, ipw_check_list = [result[1] for result in results], [result[2] for result in results]
    if errors:
        rmse_list = [result[4] for result in results]

    # Compute final estimator
    theta_0_hat = np.mean(theta_0_check_list)
    if classical:
        reg_hat, ipw_hat = np.mean(reg_check_list), np.mean(ipw_check_list)

    # Inference: estimate standard deviation and construct confidence interval
    sigma_hat = np.sqrt(np.mean((np.array(scores_list)-theta_0_hat)**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    CI = np.array([theta_0_hat-quantile*sigma_hat/np.sqrt(N), theta_0_hat+quantile*sigma_hat/np.sqrt(N)])

    # Average RMSEs across folds
    if errors:
        rmse = np.mean(rmse_list, axis=0)

    # Return results
    if classical and errors:
        return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI, rmse
    elif classical and (not errors):
        return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI
    elif (not classical) and errors:
        return theta_0_hat, sigma_hat, CI, rmse
    else:
        return theta_0_hat, sigma_hat, CI
