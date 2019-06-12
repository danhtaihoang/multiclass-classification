##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler

""" --------------------------------------------------------------------------------------
2019.06.12: Synthesize data 
Input: data length l, number of variable n, std of interactions g
Output: w[n], X[l,n],y[l] (my  = y.unique())
"""
def synthesize_data(l,n,my,g,data_type='continuous'):        
    if data_type == 'binary':
        X = np.sign(np.random.rand(l,n)-0.5)
        w = np.random.normal(0.,g,size=(n,my))
        
    if data_type == 'continuous':
        X = 2*np.random.rand(l,n)-1
        w = np.random.normal(0.,g,size=(n,my))
        
    if data_type == 'categorical':        
        from sklearn.preprocessing import OneHotEncoder
        mx = 5 # categorical number for each variables
        # initial s (categorical variables)
        s = np.random.randint(0,mx,size=(l,n)) # integer values
        onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
        X = onehot_encoder.fit_transform(s)
        w = np.random.normal(0.,g,size=(n*mx,my))    
    
    # sum_j w_ji to each i = 0
    w -= w.mean(axis=1)[:,np.newaxis]  

    # generate y based on X and w
    h = X.dot(w)
    p = np.exp(h)
    p /= p.sum(axis=1)[:,np.newaxis] 

    y = np.full(l,1000.)
    for t in range(l):    
        while y[t] == 1000:
            k0 = np.random.randint(0,my)
            if p[t,k0] > np.random.rand():
                y[t] = k0

    # Scaler X
    #X = MinMaxScaler().fit_transform(X)
    return X,y,w


