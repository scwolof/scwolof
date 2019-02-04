
import numpy as np 

"""
Method for checking that analytical gradients are correct,
using comparison with finite difference gradients.
(C) 2017 by Simon Olofsson
"""

def gradient (f, X, eps=1e-6):
    """
    Check gradients of function.
    checkgrad.gradient (f, X, eps=1e-6)

    Inputs:
        f           Function, returns Y(X), dY/dX.
                    X       [D1, ..., Dn] 
                    Y       [E1, ..., Em]
                    dY/dX   [E1, ..., Em, D1, ..., Dn]
        eps         Step size

    Outputs:
        - Default:
        d           Relative difference |dy-dh| / |dy+dh|
        dy          Analytical gradients
        dh          Numerical gradients
        flattened   d, dy and dh flattened into three columns
    """
    shape_in = X.shape
    if not isinstance(shape_in, list):
        shape_in = list(shape_in)

    # Analytical gradient
    y, dy = f(X)
    shape_out = y.shape
    if not isinstance(shape_out, list):
        shape_out = list(shape_out)
    assert list(dy.shape) == shape_out + shape_in
    dh = np.zeros( dy.shape )

    # Indices to cycle through
    inds = [ np.arange(D, dtype=int) for D in shape_in ]
    inds = np.meshgrid( *inds )
    inds = np.stack([ i.flatten() for i in inds ]).T
    # Numerical gradients using finite differences
    for ind in inds:
        # Step array
        t      = np.zeros( X.shape )
        exec("t["+','.join([str(i) for i in ind])+"] = eps")
        # Evaluate function
        yp = f(X + t)[0]
        ym = f(X - t)[0]
        # Finite difference computation
        expr = ':,' * len(shape_out) + ','.join(['%d'%i for i in ind])
        exec(r"dh["+expr+r"] = (yp - ym) / (2 * eps)")

    # Compute difference in gradients
    d = np.sqrt( (dy-dh)**2 / ((dy+dh)**2 + 1e-300) )

    flattened = np.c_[d.flatten(), dy.flatten(), dh.flatten()]
    return d, dy, dh, flattened
    



def hessian (f, X, eps = 0.0001, flatten=False):
    """
    Check Hessian of function.
    checkgrad.hessian (f, X, eps=1e-6)

    Inputs:
        f           Function, returns Y(X), d^2Y/dX^2.
                    X       [n, D] 
                    Y       [n, (E)]
                    dY/dX   [n, (E,) D, D]
        X           [n, D] input
        eps         Step size
        flatten     Return a more print-friendly result.

    Outputs:
        - Default:
        d           Relative difference |ddy-ddh| / |ddy+ddh|
        ddy         Analytical Hessian
        ddh         Numerical Hessian
        - Flattened:
        Numpy array np.c_[d.flatten(), dy.flatten(), dh.flatten()]
    """
    n, D = X.shape

    # Analytical Hessians
    y, ddy = f(X)
    assert y.shape[0] == n
    if y.ndim == 1:
        assert ddy.shape == (n,D,D)
    else:
        E = y.shape[1]
        assert ddy.shape == (n,E,D,D)

    # Numerical Hessians using finite differences
    ddh = np.zeros( ddy.shape )
    for i in range( D ):
        # Add steps to test points
        Ti      = np.zeros( X.shape )
        Ti[:,i] = eps
        yp = f(X + Ti)[0]
        ym = f(X - Ti)[0]
        ddh[...,i,i] = (yp - 2*y + ym) / eps**2

        for j in range(i+1, D):
            # Add steps to test points
            Tj      = np.zeros( X.shape )
            Tj[:,j] = eps
            ypp = f(X + Ti + Tj)[0]
            ypm = f(X + Ti - Tj)[0]
            ymp = f(X - Ti + Tj)[0]
            ymm = f(X - Ti - Tj)[0]
            ddh[...,i,j] = (ypp - ypm - ymp + ymm) / (4 * eps**2)
            ddh[...,j,i] = ddh[...,i,j]

    # Compute difference in gradients
    d = np.sqrt( (ddy-ddh)**2 / ((ddy+ddh)**2 + 1e-300) )

    if flatten: 
        return np.c_[d.flatten(), ddy.flatten(),ddh.flatten()]
    return d,ddy,ddh
