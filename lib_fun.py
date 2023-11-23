import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from sklearn.model_selection import KFold # import KFold
from sklearn.utils import resample


def wire(X, y, metric):   
    """calculate sample kernel matrix for WIRE
    INPUT
    X: n by p design matrix
    y: n by q response vector
    metric: user-defined metric, for example Euclidean, Wasserstein, Geodesic
    OUTPUT
    M: p by p matrix
    """ 
    n = X.shape[0]
    y = y.reshape((n,-1))
    x0 = X - np.mean(X, axis=0)            
    q = y.shape[1]
    if metric == "Euclidean":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.sum(np.square(temp),2))
    if metric == "Wasserstein":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.mean(np.square(temp),2))
#         indnon0 = np.nonzero(D)
        D = D/(1+D)
    if metric == "Geodesic":
        D = np.arccos(np.minimum(y@y.T, 1)) #np.arccos(y @ y.T)
    M = -(x0.T @ D @ x0)/(n**2)
    return M

def gwire(sigma, M, Nb, lam, weight= None, max_iter=100):
    """solve GWIRE using ADMM for given lambda
    INPUT
    sigma: sample covariance matrix p by p
    M: sample kernel matrix
    Nb: neighborhood information
    lam: penalty parameter
    weight: optional weight for each component of penalty term
    max_iter: optional maximum iteration number 
    OUTPUT
    B: estimated p by p matrix (sigma^{-1}Msigma^{-1})
    """
    p = sigma.shape[0]
    rho = 1
    D, P = eigh(sigma)
    C1 = np.outer(D,D)
    C = C1/(C1 + rho)
    # default ADMM paramter 
    if weight is None:
        weight = np.ones(p) 
    rho = 1
    
    # ADMM: initial values
    W = np.zeros((p,p))
    sumV = np.zeros((p,p))
    
    nblen = [len(Nb[i])>1 for i in range(len(Nb))]
    group1 = np.where( np.array(nblen) )[0]
    group2 = np.setdiff1d(range(len(Nb)), group1)
    Nbeq1 = [Nb[i][0] for i in group2]
    V = [np.zeros((len(Nb[i]), p)) for i in range(p)]

    for ite in range(max_iter):
        # print(ite)
        # update B # using V_old, W_old
        L = M/rho - W + (sumV + sumV.T)/2   
        B = L - P @ (C * (P.T @ L @ P)) @ P.T
        # assert np.allclose(B, B.T)
        # update V  # using V_old, B_new, W_old
        for i in group1:
            Ni = Nb[i]
            # print(Ni)
            Square_Ni = B[Ni,:] + W[Ni,:] - sumV[Ni,:] + V[i]
            if norm(Square_Ni)>0:
                new = max(1 - lam*weight[i]/(rho * norm(Square_Ni)), 0) * Square_Ni
            else:
                new = np.zeros((len(Ni), p))
            sumV[Ni,:] = sumV[Ni,:] - V[i] + new
            V[i] = new
            # temp[Ni,:] += new

        vs = B[Nbeq1,:] + W[Nbeq1,:]
        a = np.maximum(1 - lam*weight[Nbeq1]/(rho * norm(vs, axis = 1)), 0)
        sumV[Nbeq1,:] = (a * vs.T).T

        # update W # using B_new, V_new
        W = W + B - (sumV + sumV.T)/2
        if norm(sumV-B, 'fro') < 1e-5:
            # print('converged at', ite, 'iterations')
            break
    
    ind0 = np.where(norm(sumV, axis=1)==0)[0]
    sumV[:,ind0] = 0
    B = (sumV + sumV.T)/2
    return B

def gwire_d(X, y, metric, Nb, lam=None, weight=None, kernel = wire):
    """ estimate structral dimension d
    INPUT
    X: n by p design matrix
    y: n by q response vector
    metric: user-defined metric, for example Euclidean, Wasserstein, Geodesic
    Nb: neighborhood information
    lam: penalty parameter
    weight: weight for each component of penalty term
    kernel: function to produce kernel matrix
    OUTPUT
    d: estimated structral dimension
    """
    n,p = X.shape
    M = kernel(X,y,metric)
    sigma = np.cov(X.T)
    M_norms = np.zeros(len(Nb))
    wei = np.ones(len(Nb))
    q = 0.5
    for i, Ni in enumerate(Nb):
        wei[i] = (len(Ni))**(q)
        M_norms[i] = norm(M[Ni,:])/wei[i]
    lam_max = np.max(M_norms)
    if lam is None:
        lam = lam_max/5
    if weight is None:
        weight = wei

    B = gwire(sigma, M, Nb, lam, weight = weight, max_iter=100)
    ind = np.nonzero(norm(B, axis=1))[0]
    k = len(ind)
    value, beta_hat = eigh(B, subset_by_index=(p-k, p-1))
    beta_hat = np.fliplr(beta_hat)
    value = value[::-1]
    p2 = value/(1+np.sum(value))
    fk = np.zeros(k)
    nbs = 50
    for _ in range(nbs):    
        X_bs, y_bs = resample(X, y, replace=True)
        sigma_bs = np.cov(X_bs.T)
        M_bs = kernel(X_bs, y_bs, metric)
        Bb = gwire(sigma_bs, M_bs, Nb, lam, weight= weight, max_iter=100)
        value_b, beta_b = eigh(Bb, subset_by_index=(p-k, p-1))
        beta_b = np.fliplr(beta_b)
        value_b = value_b[::-1]
        for i in range(k-1):
            fk[i+1] += (1-np.abs(la.det(beta_hat[:,0:(i+1)].T @ beta_b[:,0:(i+1)])))/nbs
    gn = fk/(1+np.sum(fk)) + value/(1+np.sum(value))
    d = np.argmin(gn)
    return d

def gwire_cv(X, y, Nb, metric, d = None, fold=5, kernel = wire):
    """solve GWIRE using cross-validation tuned parameter 
    INPUT
    X: n by p design matrix
    y: n by q response vector
    Nb: neighborhood information
    metric:  user-defined metric, for example Euclidean, Wasserstein, Geodesic
    d: structural dimension 
    fold: number of fold for cross-validation
    kernel: function to produce kernel matrix
    OUTPUT
    beta_hat: estimated SDR directions
    d: estimated d if not given
    """  

    # using full data
    n,p = X.shape
    sigma = np.cov(X.T)
    M = kernel(X,y,metric)
    # lambda sequence for tuning
    M_norms = np.zeros(len(Nb))
    weight = np.ones(len(Nb))
    q = 0.5
    for i, Ni in enumerate(Nb):
        weight[i] = (len(Ni))**(q)
        M_norms[i] = norm(M[Ni,:])/weight[i]
    lam_max = np.max(M_norms)
    lam_seq = np.exp(np.linspace(np.log(lam_max*0.05), np.log(lam_max), num=30))
    
    if d is None:
        d = gwire_d(X, y, metric, Nb, lam=lam_max/5, weight=weight, kernel = kernel)
    # Cross validataion
    kf = KFold(n_splits = fold)
    err = np.zeros((len(lam_seq), fold))
    k = 0
    for train_index, val_index in kf.split(X):
        print('CV: Fold-', k)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # traning
        M_tr = kernel(X_train,y_train,metric)
        sigma_tr = np.cov(X_train.T)
#         D_tr, P_tr = eigh(sigma_tr)
#         C1 = np.outer(D_tr,D_tr)
#         C_tr = C1/(C1 + rho)

        # validation 
        M_val = kernel(X_val,y_val,metric)
        sigma_val = np.cov(X_val.T)
        for i, lam in enumerate(lam_seq):
            B = gwire(sigma_tr, M_tr, Nb, lam, weight)
            ind = np.nonzero(norm(B, axis=1))[0]
            s = len(ind)
            beta_hat = np.zeros((p, d))
            if s>=d:
                B_sub = B[np.ix_(ind,ind)]
                _, vec = eigh(B_sub, subset_by_index=(s-d, s-1))
                beta_hat[ind,:] = vec
            elif s>=1:
                _, beta_hat = eigh(B, subset_by_index=(p-d, p-1))
            if norm(beta_hat)!=0:
                beta_hat = beta_hat @ la.inv(sqrtm(beta_hat.T @ sigma_tr @ beta_hat))
            err[i,k] = - np.trace(beta_hat.T @ M_val @ beta_hat)
        k = k + 1

    # using 1 standard error rule choosing lambda
    err_mean = np.mean(err, axis = 1)
    err_se = np.std(err, axis = 1)/np.sqrt(fold)
    ind0 = np.argmin(err_mean)
    lam_cv = lam_seq[np.max(np.where(err_mean <= (err_mean[ind0] + err_se[ind0]) )[0])]
    lam_min = lam_seq[ind0]

    # output estimate using selected lambda
    B = gwire(sigma, M, Nb, lam_min, weight)
    ind = np.nonzero(norm(B, axis=1))[0]
    s = len(ind)
    beta_hat = np.zeros((p, d))
    if s>=d:
        B_sub = B[np.ix_(ind,ind)]
        _, vec = eigh(B_sub, subset_by_index=(s-d, s-1))
        beta_hat[ind,:] = vec
    elif s>=1:
        _, beta_hat = eigh(B, subset_by_index=(p-d, p-1))
    return beta_hat, d


# import numba
# @numba.jit(nopython=True)
def soft_thresholding(x, mu):
    """Soft-thresholding aka proximity operator of L1"""
    return np.sign(x) * np.maximum(np.abs(x) - mu, 0.)

def swire(sigma, M, d, lam, max_iter=100):
    """solve SWIRE-I using ISTA for given lambda
    INPUT
    sigma: sample covariance matrix p by p
    M: sample kernel matrix
    lam: penalty parameter
    max_iter: optional maximum iteration number 
    OUTPUT
    beta_hat: estimated p by d matrix
    """
    p = sigma.shape[1]

    # ISTA algorithm 
    t = norm(sigma, 2) ** (-2)
    B = np.zeros((p,p))
    for i in range(max_iter):
        # print(i)
        B_old = B
        temp = B - t * (sigma @ B @ sigma - M)
        B = soft_thresholding(temp, lam * t)
        # w = np.maximum(1 - lam * t / norm(temp, axis = 1), 0)
        # B = (w * temp.T).T
        if norm(B_old-B, 'fro') < 1e-5:
            # print('converged at', ite, 'iterations')
            break

    # left singular vectors for B
    beta_hat = np.zeros((p,d))
    if norm(B, 'fro') >0:
        supp_B = np.nonzero(norm(B, axis=1))[0].tolist()
        s = len(supp_B)
        if len(supp_B) >= d: # take submatrix
            B_sub = B[np.ix_(supp_B,supp_B)]
            _, U = eigh(B_sub, subset_by_index=(s-d, s-1))
            beta_hat[supp_B,:] = U
        else: # use full matrix
            _, U = eigh(B, subset_by_index=(p-d, p-1))
            beta_hat = U
    # else:
    #     beta_hat = np.zeros((p,d))
    return beta_hat

def swire_cv(X, y, metric, d, fold=5, kernel = wire):
    """solve SWIRE-I using cross-validation tuned parameter 
    INPUT
    X: n by p design matrix
    y: n by q response vector
    metric:  user-defined metric, for example Euclidean, Wasserstein, Geodesic
    d: structural dimension 
    fold: number of fold for cross-validation
    OUTPUT
    beta_hat: estimated SDR directions
    """ 
    sigma = np.cov(X.T)
    M = kernel(X,y,metric)

    # lambda sequence for tuning
    lam_max = np.max(np.abs(M))
    lam_seq = np.exp(np.linspace(np.log(lam_max*0.05), np.log(lam_max), num=30))

    # Cross validataion
    kf = KFold(n_splits = fold)
    err = np.zeros((len(lam_seq), fold))
    k = 0
    for train_index, val_index in kf.split(X):
        print('CV: Fold-', k)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # traning
        sigma_tr = np.cov(X_train.T)
        M_tr = kernel(X_train,y_train,metric)

        # validation 
        M_val = kernel(X_val,y_val,metric)
        for i, lam in enumerate(lam_seq):
            beta_lam = swire(sigma_tr, M_tr, d, lam)
            if norm(beta_lam)!=0: # normalize wrt sigma
                beta_lam = beta_lam @ la.inv(sqrtm(beta_lam.T @ sigma_tr @ beta_lam))
            err[i,k] += - np.trace(beta_lam.T @ M_val @ beta_lam)
        k = k + 1

    # using 1 standard error rule choosing lambda
    err_mean = np.mean(err, axis = 1)
    err_se = np.std(err, axis = 1)/np.sqrt(fold)
    ind0 = np.argmin(err_mean)
    lam_cv = lam_seq[np.max(np.where(err_mean <= (err_mean[ind0] + err_se[ind0]) )[0])]
    lam_min = lam_seq[ind0]
    # print(lam_cv)

    # output estimate using selected lambd
    beta_hat = swire(sigma, M, d, lam = lam_min)
    return beta_hat

def bwire(sigma, M, d, lam, max_iter=100):
    """solve SWIRE-II using ISTA for given lambda
    INPUT
    sigma: sample covariance matrix p by p
    M: sample kernel matrix
    lam: penalty parameter
    max_iter: optional maximum iteration number 
    OUTPUT
    beta_hat: estimated p by d matrix
    """
    p = sigma.shape[1]

    # ISTA algorithm 
    t = norm(sigma, 2) ** (-2)
    B = np.zeros((p,p))
    for i in range(max_iter):
        # print(i)
        B_old = B
        temp = B - t * (sigma @ B @ sigma - M)
#         B = soft_thresholding(temp, lam * t)
        w = np.maximum(1 - lam * t / norm(temp, axis = 1), 0)
        B = (w * temp.T).T
        if norm(B_old-B, 'fro') < 1e-5:
            # print('converged at', ite, 'iterations')
            break

    # left singular vectors for B
    beta_hat = np.zeros((p,d))
    if norm(B, 'fro') >0:
        supp_B = np.nonzero(norm(B, axis=1))[0].tolist()
        s = len(supp_B)
        if len(supp_B) >= d: # take submatrix
            B_sub = B[np.ix_(supp_B,supp_B)]
            _, U = eigh(B_sub, subset_by_index=(s-d, s-1))
            beta_hat[supp_B,:] = U
        else: # use full matrix
            _, U = eigh(B, subset_by_index=(p-d, p-1))
            beta_hat = U
    # else:
    #     beta_hat = np.zeros((p,d))
    return beta_hat

def bwire_cv(X, y, metric, d, fold=5, kernel = wire):
    """solve SWIRE-II using cross-validation tuned parameter 
    INPUT
    X: n by p design matrix
    y: n by q response vector
    metric:  user-defined metric, for example Euclidean, Wasserstein, Geodesic
    d: structural dimension 
    fold: number of fold for cross-validation
    OUTPUT
    beta_hat: estimated SDR directions
    """ 
    sigma = np.cov(X.T)
    M = kernel(X,y,metric)

    # lambda sequence for tuning
#     lam_max = np.max(np.abs(M))
    lam_max = np.max(norm(M, axis = 1))
    lam_seq = np.exp(np.linspace(np.log(lam_max*0.05), np.log(lam_max), num=30))

    # Cross validataion
    kf = KFold(n_splits = fold)
    err = np.zeros((len(lam_seq), fold))
    k = 0
    for train_index, val_index in kf.split(X):
        print('CV: Fold-', k)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # traning
        sigma_tr = np.cov(X_train.T)
        M_tr = kernel(X_train,y_train,metric)
        # validation 
        M_val = kernel(X_val,y_val,metric)
        for i, lam in enumerate(lam_seq):
            beta_lam = bwire(sigma_tr, M_tr, d, lam)
            if norm(beta_lam)!=0: # normalize wrt sigma
                beta_lam = beta_lam @ la.inv(sqrtm(beta_lam.T @ sigma_tr @ beta_lam))
            err[i,k] += - np.trace(beta_lam.T @ M_val @ beta_lam)
        k = k + 1

    # using 1 standard error rule choosing lambda
    err_mean = np.mean(err, axis = 1)
    err_se = np.std(err, axis = 1)/np.sqrt(fold)
    ind0 = np.argmin(err_mean)
    lam_cv = lam_seq[np.max(np.where(err_mean <= (err_mean[ind0] + err_se[ind0]) )[0])]
    lam_min = lam_seq[ind0]
    # print(lam_cv)

    # output estimate using selected lambd
    beta_hat = bwire(sigma, M, d, lam = lam_min)
    return beta_hat

def loss(A, A_hat):
    '''assessment of estimates
    INPUT
    A: true parameter
    A_hat: estimate
    OUTPUT
    general loss
    correct of detect all variables
    false positive
    false negative
    '''
    loss_general = norm(A_hat @ A_hat.T - A @ A.T)
    S = np.nonzero(norm(A, axis = 1))[0]
    S_hat = np.nonzero(norm(A_hat, axis = 1))[0]
    false_positive = len(set(S_hat).difference(S)) ## false positive
    false_negative = len(set(S).difference(S_hat)) ## false negative
    correct = (false_positive == 0 and false_negative == 0)

    return round(loss_general,3), int(correct), false_positive, false_negative

