
import numpy as np 

"""
Method for checking that analytical gradients are correct,
using comparison with finite difference gradients.
(C) 2017 by Simon Olofsson
"""

def gradient (f, X, eps=1e-6, flatten=False):
	"""
	Check gradients of function.
	checkgrad.gradient (f, X, eps=1e-6)

	Inputs:
		f 			Function, returns Y(X), dY/dX.
					X 	[n, D] 
					Y 	[n, (E)]
					dY/dX 	[n, (E,) D]
		X 			[n, D] input
		eps 		Step size
		flatten 	Return a more print-friendly result.

	Outputs:
		- Default:
		d 			Relative difference |dy-dh| / |dy+dh|
		dy 			Analytical gradients
		dh 			Numerical gradients
		- Flattened:
		Numpy array np.c_[d.flatten(), dy.flatten(), dh.flatten()]
	"""

	# Analytical gradient
	_, dy = f(X)

	# Numerical gradients using finite differences
	dh = np.zeros( dy.shape )
	for k in range( X.shape[1] ):
		# Step array
		t      = np.zeros( X.shape )
		t[:,k] = eps
		# Evaluate function
		yp = f(X + t)[0]
		ym = f(X - t)[0]
		# Finite difference computation
		dh[...,k] = (yp - ym) / (2 * eps)

	# Compute difference in gradients
	d = np.sqrt( (dy-dh)**2 / ((dy+dh)**2 + 1e-300) )

	if flatten: 
		return np.c_[d.flatten(), dy.flatten(), dh.flatten()]
	return d, dy, dh
	





def hessian (f, X, eps = 0.0001, flatten=False):
	"""
	Check Hessian of function.
	checkgrad.hessian (f, X, eps=1e-6)

	Inputs:
		f 			Function, returns Y(X), d^2Y/dX^2.
					X 		[n, D] 
					Y 		[n, (E)]
					dY/dX 	[n, (E,) D, D]
		X 			[n, D] input
		eps 		Step size
		flatten 	Return a more print-friendly result.

	Outputs:
		- Default:
		d 			Relative difference |ddy-ddh| / |ddy+ddh|
		ddy 		Analytical Hessian
		ddh 		Numerical Hessian
		- Flattened:
		Numpy array np.c_[d.flatten(), dy.flatten(), dh.flatten()]
	"""

	# Analytical Hessians
	y, ddy = f(X)

	# Numerical Hessians using finite differences
	ddh = np.zeros( ddy.shape )
	for i in range( X.shape[1] ):
		# Add steps to test points
		Ti      = np.zeros( X.shape )
		Ti[:,i] = eps
		yp = f(X + Ti)[0]
		ym = f(X - Ti)[0]
		ddh[...,i,i] = (yp - 2*y + ym) / eps**2

		for j in range(i+1, X.shape[1]):
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
