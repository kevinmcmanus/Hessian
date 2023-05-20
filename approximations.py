import numpy as np

def euler_method(dfx, x0, t):
    x0a = np.atleast_1d(x0)
    if x0a.ndim > 1:
        raise ValueError('Only 1d array or scalar allowed')
    
    nx = x0a.shape[0]
    nt = len(t)
    out = np.full((nt, nx), np.nan)

    last_x = out[0] = x0a

    for i in range(1, nt):
        dx = dfx(last_x)
        dt = t[i]-t[i-1]
        this_x = last_x + dx*dt
        last_x = out[i] = this_x
    
    return out
        
