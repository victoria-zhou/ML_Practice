# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def compute_mean_normalise(var):
    
    mean = np.mean(var)
    size = len(var)
    
    var_sum = []
    
    for i in range(size):
        
        value = var[i] - mean
        
        var_sum.append(value)
        
    return var_sum

# %%
def compute_dot_product(var_x, var_y):
    
    size = len(var_x)
    dot_sum = 0
    
    for i in range(size):   
        product = var_x[i] * var_y[i]
        dot_sum += product
    
    return dot_sum

# %%
def compute_covariance(var_x, var_y):
    
    size = len(var_x)
    
    if size != len(var_y):
        raise Exception('Sorry, two vectors need to be the same size')
    
    x_mean_norm = compute_mean_normalise(var_x)
    y_mean_norm = compute_mean_normalise(var_y)
    
    dot_sum = compute_dot_product(x_mean_norm, y_mean_norm)
    
    return dot_sum/(size - 1)

# %%
def compute_correlation(var_x, var_y):
    
    size = len(var_x)
    
    if size != len(var_y):
        raise Exception('sorry, size of var_x and var_y need to be the same')
      
    # calculate nominator of corelation
    corr_nominator = (size - 1) * compute_covariance(var_x, var_y)
    
    # calculate denominator of corelation
    var_x_mean_norm = compute_mean_normalise(var_x)
    var_y_mean_norm = compute_mean_normalise(var_y)
    
    x_sqr_sum = np.sum(np.square(var_x_mean_norm))
    y_sqr_sum = np.sum(np.square(var_y_mean_norm))
    x_y_sqr_prod = x_sqr_sum * y_sqr_sum
    
    corr_denominator = np.sqrt(x_y_sqr_prod)
    
    return corr_nominator/corr_denominator

# %%
def compute_slope(X, Y):
    
    nominator = np.mean(X) * np.mean(Y) - np.mean(X*Y)
    denominator = np.square(np.mean(X)) - np.mean(np.square(X))

    return nominator / denominator

# %%
def best_fit_params(X, Y):
    
    m_hat = compute_slope(X, Y)
    
    intercept = np.mean(Y) - m_hat * np.mean(X)
    
    return m_hat, intercept

# %%
def regresssion_line(m, c, X):
    
    y_hat = m*X + c
    
    return y_hat