# Parallelized DML algorithm
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.base import is_regressor
from scipy.stats import norm
from data_generation import g_0, m_0


# Infeasible method-of-moments estimator of the ATT
def mm_att(y_data, d_data, x_data):
    return np.mean(d_data*(y_data-g_0(0, x_data)) - m_0(x_data)*(1-d_data)*(y_data-g_0(0, x_data))/(1-m_0(x_data))) / np.mean(d_data)


# DML estimator of the ATT 
def dml_parallel_att(y_data, d_data, x_data_all, model_g0, model_m, K=5, alpha=0.05, errors=True, m_bounds=None):
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

        # Estimate outcome regression function g_0(0) and propensity score m_0 in parallel
        model_data_list = [(model_g0, x_train[d_train==0], y_train[d_train==0]), (model_m, x_train, d_train)]
        g_0_hat, m_0_hat = Parallel(n_jobs=2)(delayed(fit_predict)(model, X, y) for model, X, y in model_data_list)
        if m_bounds is not None:
            np.clip(m_0_hat, m_bounds[0], m_bounds[1], out=m_0_hat)

        # Compute auxiliary estimator and store scores for variance estimation
        scores = d_eval*(y_eval-g_0_hat) - m_0_hat*(1-d_eval)*(y_eval-g_0_hat)/(1-m_0_hat)
        theta_0_check = np.mean(scores)/np.mean(d_eval)

        # Assess RMSE of ML models on evaluation set
        if errors:
            rmse_g0 = root_mean_squared_error(g_0(0, x_eval_orig), g_0_hat)
            rmse_m = root_mean_squared_error(m_0(x_eval_orig), m_0_hat)
        else:
            rmse_g0, rmse_m = None, None

        return theta_0_check, lambda theta: (scores-theta*d_eval)/np.mean(d_train), [rmse_g0, rmse_m]

    # Partition the data for cross-fitting
    skf = StratifiedKFold(n_splits=K, shuffle=False)

    # Cross-fitting, where the different splits are processed in parallel
    results = Parallel(n_jobs=K)(delayed(process_single_split)(train_indices, eval_indices) for train_indices, eval_indices in skf.split(X=x_data, y=d_data))

    # Collect results
    theta_0_check_list = [result[0] for result in results]
    scores_list = [result[1] for result in results]
    if errors:
        rmse_list = [result[2] for result in results]

    # Compute final estimator
    theta_0_hat = np.mean(theta_0_check_list)

    # Inference: estimate standard deviation and construct confidence interval
    sigma_hat = np.sqrt(np.mean(np.concatenate([phi(theta_0_hat) for phi in scores_list])**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    CI = np.array([theta_0_hat-quantile*sigma_hat/np.sqrt(N), theta_0_hat+quantile*sigma_hat/np.sqrt(N)])

    # Average RMSEs across folds
    if errors:
        rmse = np.mean(rmse_list, axis=0)

    # Return results
    if errors:
        return theta_0_hat, sigma_hat, CI, rmse
    else:
        return theta_0_hat, sigma_hat, CI
