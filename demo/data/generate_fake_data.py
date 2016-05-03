'''
Generate data for demos and testing
'''

import pandas as pd
import numpy as np

# Create a multinomial outcome data set with several predictor variables
n_rows = 3000
n_outcomes = 3
n_vars_per_type = 10
n_types = 3
np.random.seed(10)
normals = pd.DataFrame(np.random.normal(size = (n_rows, n_vars_per_type)))
poissons = pd.DataFrame(np.random.poisson(size = (n_rows, n_vars_per_type)))
bernoullis = pd.DataFrame(np.random.binomial(1, 0.3, size = (n_rows, n_vars_per_type)))
df = pd.concat([normals, poissons, bernoullis], axis = 1)
df.columns = [str(k) for k in range(df.shape[1])]
ppp = pd.DataFrame(np.zeros((n_rows,n_outcomes)))
for pp in range(n_outcomes):
    beta0 = np.random.normal(size = n_types*n_vars_per_type)
    beta1 = np.random.normal(size = n_types*n_vars_per_type)
    beta2 = np.random.normal(size = n_types*n_vars_per_type)
    cos_coeff = np.random.normal(size = n_types*n_vars_per_type)    
    for vv in range(n_types*n_vars_per_type):
        new = (np.array(ppp[[pp]]) + beta0[vv] + beta1[vv]*df[[vv]] + 
            beta2[vv]*(df[[vv]]**2) + cos_coeff[[vv]]*np.cos(df[[vv]]))
        ppp[[pp]] = new
    ppp[[pp]] = np.exp(ppp[[pp]])
ppp = ppp.div(ppp.sum(axis = 1), axis='index')
def multinom_sampler(x):
    out = np.random.choice(a = list(range(n_outcomes)), size = 1, replace = True, p = x)
    return out
df['target'] = ppp.apply(multinom_sampler, axis = 1)[0]
df.to_csv('fake_multinomial_data.csv', index = False)
