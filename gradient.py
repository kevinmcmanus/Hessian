import numpy as np

def gradient(fx, X, eps = 1e-6):

    if X.ndim != 1:
        raise ValueError('X must be one dimensional')

    nX = X.shape[0]
    grad = np.full( nX, np.nan)

    #loop thru X, approx the deriv at each x
    for i in range(nX):
        xx = X.astype(np.float64, copy=True)
        x = xx[i]

        # look up
        x_up = x+eps
        xx[i] = x_up
        f_up = fx(xx)

        # look down
        x_down = x-eps
        xx[i] = x_down
        f_down= fx(xx)

        # calc gradient for this dim
        grad[i] = (f_up - f_down)/(x_up - x_down)

    return grad
