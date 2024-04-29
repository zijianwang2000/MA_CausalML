# Data-generating process
import numpy as np
from scipy.stats import t


mean = np.linspace(0.7, 0.0, 8)
cov = np.array([[round(0.6**abs(i-j)*((-1.01)**(i+j)), 3) for j in range(8)] for i in range(8)])
beta = np.linspace(-0.8, 1.0, 10)
df = 10
gamma = np.array([1.0, 2.0, 2.0, 3.0])
F = lambda z: 1.0 / (1.0 + np.exp(-z))   # standard logistic function

# Propensity score
m_0 = lambda x: t.cdf(x @ beta, df)

# Outcome regression function
def g_0(d, x):
    if x.ndim == 1:
        x = x.reshape(1,-1)
    linear_part = x[:,:4] @ gamma + x[:,4]*(d+1)
    nonlinear_part = F(x[:,5])*x[:,6]**2 - x[:,8]*(np.sqrt(x[:,9])+2*x[:,6]) + d*x[:,2]*x[:,8]**(3/2)
    return linear_part + nonlinear_part


# Generate a data set of size N in vectorized fashion
def get_data(N, rng):
    x_normal = rng.multivariate_normal(mean=mean, cov=cov, size=N)
    x_uniform = rng.uniform(size=(N,2))
    x_data = np.concatenate((x_normal, x_uniform), axis=1)

    xi = rng.standard_t(df=df, size=N)
    d_data = (x_data @ beta + xi >= 0).astype(float)
        
    u = rng.normal(scale=np.mean(np.abs(x_data), axis=-1))
    y_data = g_0(d_data, x_data) + u 

    return y_data, d_data, x_data
    