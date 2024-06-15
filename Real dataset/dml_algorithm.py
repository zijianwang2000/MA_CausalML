# DML algorithm
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.base import is_regressor
from scipy.stats import norm


# DML estimator of the ATE
def dml_parallel_ate(y_data, d_data, x_data, model_g, model_m, K=5, alpha=0.05, classical=False, m_bounds=None):
    # Process one data split in the cross-fitting procedure
    def process_single_split(train_indices, eval_indices):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_eval, d_eval, x_eval = y_data[eval_indices], d_data[eval_indices], x_data[eval_indices]

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

        return theta_0_check, reg_check, ipw_check, scores

    # Partition the data for cross-fitting
    skf = StratifiedKFold(n_splits=K, shuffle=False)

    # Cross-fitting, where the different splits are processed in parallel
    results = Parallel(n_jobs=K)(delayed(process_single_split)(train_indices, eval_indices) for train_indices, eval_indices in skf.split(X=x_data, y=d_data))

    # Collect results (in particular the auxiliary estimators)
    theta_0_check_list = [result[0] for result in results]
    scores_list = [result[3] for result in results]   # Needed for variance estimation
    if classical:
        reg_check_list, ipw_check_list = [result[1] for result in results], [result[2] for result in results]

    # Compute final estimator
    theta_0_hat = np.mean(theta_0_check_list)
    if classical:
        reg_hat, ipw_hat = np.mean(reg_check_list), np.mean(ipw_check_list)

    # Inference: estimate standard deviation and construct confidence interval
    sigma_hat = np.sqrt(np.mean((np.concatenate(scores_list)-theta_0_hat)**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    CI = np.array([theta_0_hat-quantile*sigma_hat/np.sqrt(N), theta_0_hat+quantile*sigma_hat/np.sqrt(N)])

    # Return results
    if classical:
        return np.array([theta_0_hat, reg_hat, ipw_hat]), sigma_hat, CI
    else:
        return theta_0_hat, sigma_hat, CI
