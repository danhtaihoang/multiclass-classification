##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
#=========================================================================================
# 2019.06.112: Expectation Reflection for multiclass classification
# input: x[l,n], y[l]
# output: h0, w[n,m] (m = y.unique())
def fit(x,y,niter_max=50,lamda=0.0):
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    y = onehot_encoder.fit_transform(y.reshape(-1,1))

    mx = x.shape[1]
    my = y.shape[1]

    y2 = 2*y-1

    x_av = x.mean(axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    # 2019.05.15: ridge regression
    c += lamda*np.identity(mx)
    c_inv = linalg.pinvh(c)

    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)

    cost = np.full(niter_max,100.)         
    for iloop in range(niter_max):
        h = h0[np.newaxis,:] + x.dot(w)
        p = np.exp(h)
        
        # normalize
        p_sum = p.sum(axis=1)       
        p /= p_sum[:,np.newaxis]        
        h = np.log(p)
        
        #p2 = p_sum[:,np.newaxis] - p
        p2 = 1. - p
        h2 = np.log(p2)

        hh2 = h-h2
        model_ex = np.tanh(hh2/2)

        cost[iloop] = ((y2 - model_ex)**2).mean()
        if iloop > 0 and cost[iloop] >= cost[iloop-1]: break
        #print(cost[iloop])

        t = hh2 !=0    
        h[t] = h2[t] + y2[t]*hh2[t]/model_ex[t]
        h[~t] = h2[~t] + y2[~t]*2

        h_av = h.mean(axis=0)
        dh = h - h_av

        dhdx = dh[:,np.newaxis,:]*dx[:,:,np.newaxis]
        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)            

        w -= w.mean(axis=0) 

        h0 = h_av - x_av.dot(w)
        h0 -= h0.mean()
 
    return h0,w #,cost,iloop

""" --------------------------------------------------------------------------------------
2019.06.12: calculate probability p based on x,h0, and w
input: x[l,n], w[n,my], h0
output: p[l]
"""
def predict(x,h0,w):
    h = h0[np.newaxis,:] + x.dot(w)
    p = np.exp(h)
        
    # normalize
    p_sum = p.sum(axis=1)       
    p /= p_sum[:,np.newaxis]  

    return np.argmax(p,axis=1)
