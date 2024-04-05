# Data-generating process
import numpy as np


mean = np.array([1.0, 0.0])
cov = np.array([[1.0, -0.2], [-0.2, 0.5]])
beta = np.array([1.0, 2.0, -1.0])
F = lambda z: 1.0 / (1.0 + np.exp(-z))   # standard logistic function

# Propensity score
m_0 = lambda x: F(x @ beta)

# Outcome regression function
def g_0(d, x):
    if x.ndim == 1:
        x = x.reshape(1,-1)
    return d*x[:,0] + F(x[:,1]) - 2*x[:,2]**2


# Generate a data set of size N in vectorized fashion
def get_data(N):
    x_12 = np.random.multivariate_normal(mean=mean, cov=cov, size=N)
    x_3 = np.random.uniform(size=N)
    x_data = np.concatenate((x_12, x_3.reshape(N,1)), axis=1)

    xi = np.random.logistic(size=N)
    d_data = (x_data @ beta + xi >= 0).astype(float)
        
    u = np.random.normal(scale=x_3)
    y_data = g_0(d_data, x_data) + u 

    return y_data, d_data, x_data
