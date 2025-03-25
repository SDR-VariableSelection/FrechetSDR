
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy import stats
from sklearn.model_selection import KFold # import KFold
from sklearn.utils import resample
from lib_fun import *
#%%
hour = pd.read_csv("hour.csv")
index = [hour.dteday[i] + "-" + str(x) for i, x in enumerate(list(hour.hr))]
index[0:24]
hour.index = pd.to_datetime(index, format='%Y-%m-%d-%H')
hour.shape

df = hour.pivot(index='dteday',columns='hr')[['cnt']]
df=df.fillna(0)
old = df.columns.values.tolist()
new = ['{}'.format(x) for x in range(0, 24)]
df = df.set_axis(new, axis=1)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.drop(index="2012-10-29")
df.shape

day = pd.read_csv("day.csv")
date = day.dteday
day.index = pd.to_datetime(day.dteday, format='%Y-%m-%d')
Xdat = day.drop(columns=["instant","dteday","season","yr","mnth","weekday","casual","registered","cnt"])
Xdat["BW"] = (Xdat.weathersit == 2).astype(int)        
Xdat["RBW"] = (Xdat.weathersit > 2).astype(int)   
Xdat = Xdat.drop(columns = ["weathersit"])
Xdat = Xdat.drop(index="2012-10-29")
Xdat.head()

Xdat = pd.concat(
   [
       Xdat,
       pd.DataFrame(
           np.random.randn(730, 1000)*0.1, 
           index=df.index, 
           columns= ['X{}'.format(x) for x in range(0, 1000)]
       )
   ], axis=1
)
Xdat.head()
Xdat.columns[0:8]

#%%
df_density = df
for i in range(df.shape[0]):
    df_density.iloc[i,:] = df.iloc[i,:]/np.sum(df.iloc[i,:])
dfsqr = np.sqrt(df_density)/np.sqrt(2)
Y = np.array(dfsqr)
X = np.array(Xdat)
X.shape
Y = np.array(dfsqr)
Y.shape
Xs = (X - np.mean(X, axis = 0))/ np.std(X, axis = 0)
#Xs = X.copy()
Ys = Y.copy()
#%%
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
model = GraphicalLassoCV()
model.fit(Xs[:,:8])
model.alpha_
# cov_ = model.covariance_
ome = model.precision_
np.where(np.sum(ome!=0, axis = 1) > 1)[0]
p = X.shape[1]
omega = np.eye(p)
Nb =[]
for i in range(8):
    for j in range(8):
        if abs(ome[i,j]) > 0.01:
            omega[i,j] = 1
for i in range(p):
    Ni = (np.nonzero(omega[i,:])[0]).tolist() #np.nonzero(ome[i,:])[0]
    Nb.append(Ni)

# sigma = np.cov(Xs.T) 
metric = "Euclidean"
# M = wire(X, Y, metric)
# %%
# d_est = gwire_d(Xs, Ys, metric, Nb)
beta_g, _ = gwire_cv(Xs, Ys, Nb, metric, d = 2, fold=5)
supp_hat = np.nonzero(norm(beta_g, axis=1))[0].tolist()
beta_s = swire_cv(Xs, Ys, metric, d = 2)
beta_b = bwire_cv(Xs, Ys, metric, d = 2)
beta_g = np.fliplr(beta_g)
beta_s = np.fliplr(beta_s)
beta_b = np.fliplr(beta_b)
#%% plots
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm

Dir1 = Xs @ beta_g[:,0]
Dir2 = Xs @ beta_g[:,1]

#%% First Sufficient Predictor
ind = np.argsort(Dir1)
seq = np.linspace(0,700,101,endpoint=True)
seq = seq.astype(int)
subind = ind[seq]
subind

n = X.shape[0]
fig = plt.figure(figsize =(10,10))
# fig.subplots_adjust(bottom=-0.15,top=1.2)

ax = plt.axes(projection ='3d')
# ax.set_box_aspect(aspect = (15,15,30))

for i in subind:
  x = [Dir1[i]] * 1000 # first sufficient directions
  y = np.arange(24) # hours
  z = Y[i,:] # density, replace by y[i]
  B_spline_coeff = make_interp_spline(y, z)
  y_Final = np.linspace(y.min(), y.max(), 1000)
  z_Final = B_spline_coeff(y_Final)
  ax.scatter(x,y_Final,z_Final, c =z_Final, cmap = 'rainbow', s = 0.1)

ax.set_xlabel("First Sufficient Predictor", fontsize=15)
ax.set_ylabel("Hours", fontsize=15)
ax.set_zlabel("Counts", fontsize=15)
ax.view_init(30, 200)
fig.tight_layout()

#%% Second Sufficient Predictor
ind1 = np.where(Xdat["workingday"] == 0)[0] # Working days or Xdat["workingday"] == 0 for Non-Workding days
len(ind1)
seq = np.linspace(0,len(ind1),100,endpoint=False)
# subind = ind[range(100)*7]
seq = seq.astype(int)
subind = ind1[seq]
subind

n = X.shape[0]
fig = plt.figure(figsize =(10,10))
ax = plt.axes(projection ='3d')
for i in subind:
  x = [Dir2[i]] * 1000 # first sufficient directions
  y = np.arange(24) # hours
  z = Y[i,:] # density, replace by y[i]
  B_spline_coeff = make_interp_spline(y, z)
  y_Final = np.linspace(y.min(), y.max(), 1000)
  z_Final = B_spline_coeff(y_Final)
  ax.scatter(x,y_Final,z_Final, c =z_Final, cmap = 'rainbow', s = 0.1)

ax.set_xlabel("Second Sufficient Predictor", fontsize=15)
ax.set_ylabel("Hours", fontsize=15)
ax.set_zlabel("Counts", fontsize=15)
#%%
ax.view_init(30, 250)
fig.tight_layout()

plt.plot(Dir2[subind], np.max(Y[subind,:], axis = 1), 'r.')
plt.xlabel("Second Sufficient Predictor")
plt.ylabel("Maximum Counts")


# %%
