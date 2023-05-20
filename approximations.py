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
    if nx == 1:
        out = np.squeeze(out)
    return out.T
        
def runge_kutter2_method(dfx, x0, t):
    x0a = np.atleast_1d(x0)
    if x0a.ndim > 1:
        raise ValueError('Only 1d array or scalar allowed')
    
    nx = x0a.shape[0]
    nt = len(t)
    out = np.full((nt, nx), np.nan)

    last_x = out[0] = x0a

    for i in range(1, nt):
        dx1 = dfx(last_x) #how to handle optional t param?
        h = t[i]- t[i-1]
        dx2 = dfx(last_x+(h/2)*dx1)
        this_x = last_x + h*dx2
        last_x = out[i] = this_x

    if nx == 1:
        out = np.squeeze(out)
    return out.T

def runge_kutter4_method(dfx, x0, t):
    x0a = np.atleast_1d(x0)
    if x0a.ndim > 1:
        raise ValueError('Only 1d array or scalar allowed')
    
    nx = x0a.shape[0]
    nt = len(t)
    out = np.full((nt, nx), np.nan)

    last_x = out[0] = x0a

    for i in range(1, nt):
        h = t[i]- t[i-1]
        dx1 = dfx(last_x) #how to handle optional t param?
        dx2 = dfx(last_x+(h/2)*dx1)
        dx3 = dfx(last_x+(h/2)*dx2)
        dx4 = dfx(last_x+ h*dx3)

        this_x = last_x + (h/6.)*(dx1 + 2.*dx2 + 2.*dx3 + dx4)

        last_x = out[i] = this_x
        
    if nx == 1:
        out = np.squeeze(out)
    return out.T

