# Parallelized DML algorithm
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.base import is_regressor
from scipy.stats import norm


# DML estimator of the ATE and ATT
def dml_ate_att(y_data, d_data, x_data, model_g, model_m, K=5, alpha=0.05, m_bounds=None):
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

        # Compute auxiliary estimators and store scores for variance estimation
        ate_scores = g_0_hat[1] - g_0_hat[0] + d_eval*(y_eval-g_0_hat[1])/m_0_hat - (1-d_eval)*(y_eval-g_0_hat[0])/(1-m_0_hat)
        att_scores = d_eval*(y_eval-g_0_hat[0]) - m_0_hat*(1-d_eval)*(y_eval-g_0_hat[0])/(1-m_0_hat)
        ate_aux, att_aux = np.mean(ate_scores), np.mean(att_scores)/np.mean(d_eval)

        return ate_aux, att_aux, ate_scores, lambda theta: (att_scores-theta*d_eval)/np.mean(d_train)

    # Partition the data for cross-fitting
    skf = StratifiedKFold(n_splits=K, shuffle=False)

    # Cross-fitting, where the different splits are processed in parallel
    results = Parallel(n_jobs=K)(delayed(process_single_split)(train_indices, eval_indices) for train_indices, eval_indices in skf.split(X=x_data, y=d_data))

    # Collect results
    ate_aux_list = [result[0] for result in results]
    att_aux_list = [result[1] for result in results]
    ate_scores_list = [result[2] for result in results]
    att_scores_list = [result[3] for result in results]

    # Compute final estimators
    ate_hat, att_hat = np.mean(ate_aux_list), np.mean(att_aux_list)

    # Inference: estimate standard deviation and construct confidence interval
    ate_sigma_hat = np.sqrt(np.mean((np.concatenate(ate_scores_list)-ate_hat)**2))
    att_sigma_hat = np.sqrt(np.mean(np.concatenate([phi(att_hat) for phi in att_scores_list])**2))
    N = len(y_data)
    quantile = norm.ppf(1-alpha/2)
    ate_CI = np.array([ate_hat-quantile*ate_sigma_hat/np.sqrt(N), ate_hat+quantile*ate_sigma_hat/np.sqrt(N)])
    att_CI = np.array([att_hat-quantile*att_sigma_hat/np.sqrt(N), att_hat+quantile*att_sigma_hat/np.sqrt(N)])

    # Return results
    return [ate_hat, ate_sigma_hat, ate_CI], [att_hat, att_sigma_hat, att_CI]
