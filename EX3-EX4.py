"""
Frechet Sufficient Dimension Reduction
Graphical Weighted Inverse Regression Ensemble
Example 3-4: response Y is univariate. 
"""
#%%
import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy import stats
from sklearn.model_selection import KFold # import KFold
from sklearn.utils import resample
import time
from lib_fun import *
## problem setting
#  1. EX=3 for example 3 ; EX=4 for example 4 (Table 6)

# import sys
# seedid = int(sys.argv[1])
# n = int(sys.argv[2])
# p = int(sys.argv[3])
# np.random.seed(seedid)
n, p, seedid= 500, 1000, 123
EX = 3
covstruc = 1 # or 2
    
#%%
# n, p, s = 600, 800, 10
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
Nb = []
for i in range(p):
    Ni = (np.nonzero(omega[i,:])[0]).tolist() #np.nonzero(ome[i,:])[0]
    Nb.append(Ni)

if EX == 3:
    ## EX 3
    d = 1
    beta = np.zeros((p,d))
    beta[index, 0] = 1
    u = np.random.randn(n)
    y = np.zeros((n, 1))
    y[:,0] = np.exp(X @ beta[:,0] + 0.5* u)
elif EX == 4:
    ## EX 4
    d = 2
    beta = np.zeros((p,d))
    beta[index[:groupno], 0] = 1
    beta[index[groupno:], 1] = 1
    u = np.random.randn(n)
    y = np.zeros((n, 1))
    y[:,0] = np.exp(X @ beta[:,0]) * np.sign(X @ beta[:,1]) + 0.2 * u

beta_true = beta @ sqrtm(la.inv(beta.T @ beta))
supp = np.nonzero(norm(beta_true, axis=1))[0].tolist()

#%%
import time
start = time.time()
metric = "Euclidean"
d_est0 = gwire_d(X, y, metric, Nb)
print('GWIRE estimated structure dimension:',d_est0)
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))
#%%
start = time.time()
beta_gwire, _ = gwire_cv(X, y, Nb, metric, d, fold=5)
supp_hat = np.nonzero(norm(beta_gwire, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('gwire loss:', norm(beta_gwire @ beta_gwire.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
metric = 10
d_est1 = gwire_d(X, y, metric, Nb, kernel = sir)
print('GSIR estimated structure dimension:',d_est1)
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
start = time.time()
beta_gsir, _ = gwire_cv(X, y, Nb, metric, d, fold=5, kernel=sir)
supp_hat = np.nonzero(norm(beta_gsir, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('gsir loss:', norm(beta_gsir @ beta_gsir.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
metric = None
d_est2 = gwire_d(X, y, metric, Nb, kernel = cume)
print('GCUME estimated structure dimension:',d_est2)
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))


#%%
start = time.time()
beta_gcume, _ = gwire_cv(X, y, Nb, metric, d, fold=5, kernel=cume)
supp_hat = np.nonzero(norm(beta_gcume, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('gcume loss:', norm(beta_gcume @ beta_gcume.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
metric = 'r'
d_est3 = gwire_d(X, y, metric, Nb, kernel = pHd)
print('GpHd estimated structure dimension:',d_est3)
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
start = time.time()
beta_gphd, _ = gwire_cv(X, y, Nb, metric, d, fold=5, kernel=pHd)
supp_hat = np.nonzero(norm(beta_gphd, axis=1))[0].tolist()
print(supp, supp_hat)
print('support recovery:', np.all(supp_hat == supp))
print('gpHd loss:', norm(beta_gphd @ beta_gphd.T - beta_true @ beta_true.T))
end = time.time()
print('time used:n,p =', n, p, 'is', np.round(end - start,1))

#%%
import pandas as pd
critable = pd.DataFrame()
critable.loc[0, "seedid"] = seedid
out1 = loss(beta_true, beta_gwire)
critable.loc[0,"d-est-gwire"] = d_est0
critable.loc[0,"gwire-GeneralLoss"] = out1[0]
critable.loc[0,"gwire-correct"]= out1[1]
critable.loc[0,"gwire-false_positive"] = out1[2]
critable.loc[0,"gwire-false_negative"] = out1[3]
out2 = loss(beta_true, beta_gsir)
critable.loc[0,"d-est-gsir"] = d_est1
critable.loc[0,"gsir-GeneralLoss"] = out2[0]
critable.loc[0,"gsir-correct"]= out2[1]
critable.loc[0,"gsir-false_positive"] = out2[2]
critable.loc[0,"gsir-false_negative"] = out2[3]
out3 = loss(beta_true, beta_gcume)
critable.loc[0,"d-est-gcume"] = d_est2
critable.loc[0,"gcume-GeneralLoss"] = out3[0]
critable.loc[0,"gcume-correct"]= out3[1]
critable.loc[0,"gcume-false_positive"] = out3[2]
critable.loc[0,"gcume-false_negative"] = out3[3]
out4 = loss(beta_true, beta_gphd)
critable.loc[0,"d-est-gphd"] = d_est3
critable.loc[0,"gphd-GeneralLoss"] = out4[0]
critable.loc[0,"gphd-correct"]= out4[1]
critable.loc[0,"gphd-false_positive"] = out4[2]
critable.loc[0,"gphd-false_negative"] = out4[3]
print(critable)
# %%
# t = time.strftime("%y%m%d")
# fname = f'Univariate_n{n}_p{p}_d{d}_s{s}_sigma1_{t}.csv'
# print(fname)
# import os
# if os.path.exists(fname):
#     original = pd.read_csv(fname)
#     critable = pd.concat([original,critable])
# critable.to_csv(fname,index=False)
# %%