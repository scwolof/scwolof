
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
    assert list(dy.shape) == shape_out + shape_in,\
        '%s != %s + %s' %(list(dy.shape), shape_out, shape_in)
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
    



def hessian (f, X, eps=1e-6):
    """
    Check Hessian of function.
    checkgrad.hessian (f, X, eps=1e-6)

    Inputs:
        f           Function, returns Y(X), d^2Y / dX^2.
                    X           [D1, ..., Dn] 
                    Y           [E1, ..., Em]
                    d^2Y/dX^2   [E1, ..., Em, D1, ..., Dn, D1, ..., Dn]
        eps         Step size

    Outputs:
        - Default:
        d           Relative difference |ddy-ddh| / |ddy+ddh|
        ddy         Analytical Hessian
        ddh         Numerical Hessian
        flattened   d, ddy and ddh flattened into three columns
    """
    shape_in = X.shape
    if not isinstance(shape_in, list):
        shape_in = list(shape_in)

    # Analytical Hessian
    y, ddy = f(X)
    shape_out = y.shape
    if not isinstance(shape_out, list):
        shape_out = list(shape_out)
    assert list(ddy.shape) == shape_out + shape_in + shape_in,\
        '%s != %s + %s + %s' %(list(ddy.shape), shape_out, shape_in, shape_in)
    ddh = np.zeros( ddy.shape )

    # Indices to cycle through
    inds = [ np.arange(D, dtype=int) for D in shape_in ]
    inds = np.meshgrid( *inds )
    inds = np.stack([ i.flatten() for i in inds ]).T
    # Numerical gradients using finite differences
    for j, i1 in enumerate(inds):
        expr1 = ['%d'%i for i in i1]
        # Add steps to test points
        Ti = np.zeros( X.shape )
        exec("Ti[" + ','.join(expr1) + "] = eps")
        # Evaluate function
        yp = f(X + Ti)[0]
        ym = f(X - Ti)[0]
        # Finite difference computation
        lexpr = ':,' * len(shape_out) + ','.join(expr1) + ',' + ','.join(expr1)
        #st()
        exec(r"ddh[" + lexpr + r"] = (yp - 2*y + ym) / eps**2")

        for i2 in inds[j+1:]:
            expr2 = ['%d'%i for i in i2]
            # Step array
            Tj = np.zeros( X.shape )
            exec("Tj[" + ','.join(expr2) + "] = eps")
            # Evaluate function
            ypp = f(X + Ti + Tj)[0]
            ypm = f(X + Ti - Tj)[0]
            ymp = f(X - Ti + Tj)[0]
            ymm = f(X - Ti - Tj)[0]
            # Finite difference computation
            lexpr1 = ':,' * len(shape_out) + ','.join(expr1) + ',' + ','.join(expr2)
            lexpr2 = ':,' * len(shape_out) + ','.join(expr2) + ',' + ','.join(expr1)
            exec(r"ddh[" + lexpr1 + r"] = (ypp - ypm - ymp + ymm) / (4 * eps**2)")
            exec(r"ddh[" + lexpr2 + r"] = ddh[" + lexpr1 + r"]")

    # Compute difference in gradients
    d = np.sqrt( (ddy-ddh)**2 / ((ddy+ddh)**2 + 1e-300) )

    flattened = np.c_[d.flatten(), ddy.flatten(), ddh.flatten()]
    return d, ddy, ddh, flattened
