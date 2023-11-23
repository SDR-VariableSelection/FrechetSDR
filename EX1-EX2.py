"""
Frechet Sufficient Dimension Reduction
Graphical Weighted Inverse Regression Ensemble
Example 1 and 2: response Y is distribution or on the sphere. 
"""

#%%
# packages
import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import sqrtm
from scipy import stats
from lib_fun import *
#%%
## problem setting
# 1. seedid = 1-100
# 2. EX=1 for example 1 (Table 1); EX=2 for example 2 (Table 2)
# 3. covstruc = 1 or 2 to produce Table 3
# 4. neigh = False to produce Table 5
# 5. Table 4 is based on partial results.
n,p,seedid = 500,1000,123
EX = 2 # or 2
neigh = True # or False
covstruc = 1 # or 2

#%% Generate Data
s = 10
index = range(s)
groupno = 5
nog = 5

rng = np.random.RandomState(seedid)
# true covariance matrix 
covx = np.eye(p)
covx[:groupno*nog,:groupno*nog] = np.kron(np.eye(nog), np.ones((groupno, groupno)) + .16*np.eye(groupno))
omega = la.inv(covx)
if covstruc == 2:
    omega[4,5] = 0.1
    omega[5,4] = 0.1
    omega[9,10] = 0.1
    omega[10,9] = 0.1
covx = la.inv(omega)
X = rng.multivariate_normal(np.zeros(p), covx, size = n)
sigma = np.cov(X.T)
## Neighborhood 
if not neigh:
    # Neighborhood is unknown
    # Friedman et al, “Sparse inverse covariance estimation with the graphical lasso”, Biostatistics 9, pp 432, 2008
    from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
    model = GraphicalLassoCV()
    model.fit(X)
    print(model.alpha_)
    omega = model.precision_
    np.where(np.sum(omega!=0, axis = 1) > 1)[0]

Nb = []
for i in range(p):
    Ni = (np.nonzero(omega[i,:])[0]).tolist() #np.nonzero(ome[i,:])[0]
    Nb.append(Ni)

if EX == 1:
    d = 1
    L = 1000
    beta = np.zeros((p,d))
    beta[:10,0] = 1
    mu = (X @ beta[:,0])
    y = np.zeros((n, L-1))
    meanshift = np.zeros(n)
    for i in range(n):
        meanshift[i] = 0.1* np.random.randn(1) + mu[i]
        for j in range(L-1):
            y[i,j] = meanshift[i] + 1* stats.norm.ppf((j+1)/L)
    beta_true = beta @ sqrtm(la.inv(beta.T @ beta)) # normalize wrt identity
    metric = "Wasserstein"
elif EX == 2:
    d = 2
    beta = np.zeros((p,d))
    beta[index[:groupno], 0] = 1
    beta[index[groupno:], 1] = 1
    beta_true = beta @ sqrtm(la.inv(beta.T @ beta))
    y = np.zeros((n, 4))
    u = np.random.randn(n) * 0.1
    v = 0.2
    a1 = np.sin(v * (X+1) @ beta[:,0])
    a2 = np.cos(v * (X+1) @ beta[:,0])
    b1 = np.sin(v * (X+1) @ beta[:,1])
    b2 = np.cos(v * (X+1) @ beta[:,1])
    y[:,0] = np.cos(u) * a1 * b1
    y[:,1] = np.cos(u) * a1 * b2
    y[:,2] = np.cos(u) * a2
    y[:,3] = np.sin(u)                   
    metric = "Geodesic"
# M = wire(X,y,metric)
#%%
import time
start = time.time()
d_est = gwire_d(X, y, metric, Nb)
print('estimated structure dimension:',d_est)
beta_gwire, _ = gwire_cv(X, y, Nb, metric, d, fold=5)
supp = np.nonzero(norm(beta_true, axis=1))[0].tolist()
supp_hat = np.nonzero(norm(beta_gwire, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('gwire loss:', norm(beta_gwire @ beta_gwire.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
start = time.time()
beta_swire = swire_cv(X, y, metric, d, fold=5)
supp_hat = np.nonzero(norm(beta_swire, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('swire loss:', norm(beta_swire @ beta_swire.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
start = time.time()
beta_bwire = bwire_cv(X, y, metric, d, fold=5)
supp_hat = np.nonzero(norm(beta_bwire, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('swire loss:', norm(beta_bwire @ beta_bwire.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
import pandas as pd
critable = pd.DataFrame()
critable.loc[0,"seedid"] = seedid
out1 = loss(beta_true, beta_gwire)
critable.loc[0,"d-est"] = d_est
critable.loc[0,"gwire-GeneralLoss"] = out1[0]
critable.loc[0,"gwire-correct"]= out1[1]
critable.loc[0,"gwire-false_positive"] = out1[2]
critable.loc[0,"gwire-false_negative"] = out1[3]
out2 = loss(beta_true, beta_swire)
critable.loc[0,"swireI-GeneralLoss"] = out2[0]
critable.loc[0,"swireI-correct"]= out2[1]
critable.loc[0,"swireI-false_positive"] = out2[2]
critable.loc[0,"swireI-false_negative"] = out2[3]
out3 = loss(beta_true, beta_bwire)
critable.loc[0,"swireII-GeneralLoss"] = out3[0]
critable.loc[0,"swireII-correct"]= out3[1]
critable.loc[0,"swireII-false_positive"] = out3[2]
critable.loc[0,"swireII-false_negative"] = out3[3]
print(critable)
# %%
# t = time.strftime("%y%m%d")
# fname = f'Density_n{n}_p{p}_d{d}_s{s}_sigma1_{t}.csv'
# print(fname)
# import os
# if os.path.exists(fname):
#     original = pd.read_csv(fname)
#     critable = pd.concat([original,critable])
# critable.to_csv(fname,index=False)
