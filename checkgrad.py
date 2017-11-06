
import numpy as np 

"""
Method for checking that analytical gradients are correct,
using comparison with finite difference gradients.
(C) 2017 by Simon Olofsson
"""

def check_gradient (f, X, grad=None, eps = 0.0001, flatten=False):
	"""
	Check gradients of function.
	check_gradient (f, X, grad=None, eps = 0.0001)

	Inputs:
	f 		Function, either returns f(X) or f(X),df(X).
			If it does not return gradients already,
			input parameter grad should point to
			gradient function. The function should
			take (n,D)-array inputs, and return
			n-vector function values and (n,D)-array 
			gradient values. Number of points n
			can change, but D should be fixed for 
			the function.
	X 		(n,D)-array, with n the number of data
			points and D the different dimensions
	grad 	Gradient function, only used if f does
			not already return gradients. Should
			take (n,D)-array input and return
			(n,D)-array gradient values
	eps 	Step size
	flatten Return a more print-friendly result.
			Default value is False. 

	Outputs:
	d 		Difference in analytical and numerical
			gradient values: sqrt((dy-dh)**2/(dy+dh)**2)
	- Default:
	dy 		Analytical gradients
	dh 		Numerical gradients
	- Flattened:
	dydh 	Numpy array np.c_[dy.flatten(),dh.flatten()]
	"""

	n,D = X.shape[0], 1 if X.ndim == 1 else X.shape[1]
	epsVec = eps*np.ones(n)
	zeroArray = np.zeros((n,D),dtype='float')

	# Analytical gradient
	if grad is None:
		_,gradA = f(X)
	else:
		gradA = grad(X)

	# Numerical gradients using finite differences
	gradN = zeroArray.copy()
	for k in range(D):
		# Step array
		if D == 1:
			testarray = epsVec.copy().reshape(X.shape)
		else:
			testarray = zeroArray.copy()
			testarray[:,k] = epsVec
		# Add steps to test points
		Xp, Xm = X + testarray, X - testarray
		if np.any(Xp.shape != X.shape) or np.any(Xm.shape != X.shape):
			raise ValueError('Xp or Xm has illegal shape')
		# Evaluate function
		yp,ym = [f(Xp),f(Xm)] if grad is not None else [f(Xp)[0],f(Xm)[0]]
		# Finite difference computation
		gradN[:,k] = (yp-ym)/(2*eps)

	# Compute difference in gradients
	gradDiff = np.sqrt((gradA-gradN)**2/(gradA+gradN)**2)

	if not flatten: return gradDiff,gradA,gradN
	else: return gradDiff,np.c_[gradA.flatten(),gradN.flatten()]





def check_hessian (f, X, hess=None, eps = 0.0001, flatten=False):
	"""
	Check Hessian of function.
	check_hessian (f, X, grad=None, eps = 0.0001)

	Inputs:
	f 		Function, either returns f(X) or f(X),ddf(X).
			If it does not return Hessians already,
			input parameter hess should point to
			Hessian function. The function should
			take (n,D)-array inputs, and return (n,)-
			vector function values and (n,D,D)-array 
			Hessian values. Number of points n
			can change, but D should be fixed for 
			the function.
	X 		(n,D)-array, with n the number of data
			points and D the different dimensions
	hess 	Hessian function, only used if f does
			not already return Hessians. Should
			take (n,D)-array input and return
			(n,D,D)-array Hessian values.
	eps 	Step size.
	flatten Return a more print-friendly result.
			Default value is False. 

	Outputs:
	d 		Difference in analytical and numerical
			Hessian values: sqrt((ddy-ddh)**2/(ddy+ddh)**2)
	- Default:
	ddy 	Analytical Hessians
	ddh 	Numerical Hessians
	- Flattened:
	ddyddh 	Numpy array np.c_[ddy.flatten(),ddh.flatten()]
	"""

	n,D = X.shape[0], 1 if X.ndim == 1 else X.shape[1]
	epsVec = eps*np.ones(n)
	eps2 = eps**2

	# Analytical Hessians
	if hess is None:
		fc,hessA = f(X)
	else:
		fc,hessA = f(X),grad(X)

	def get_step_array (k):
		testarray = np.zeros(n*D,dtype='float').reshape((n,D))
		testarray[:,k] = epsVec
		return testarray

	# Numerical Hessians using finite differences
	hessN = np.zeros(n*D*D,dtype='float').reshape((n,D,D))
	for i in range(D):
		# Add steps to test points
		Ti = get_step_array(i)
		Xp, Xm = X + Ti, X - Ti
		if np.any(Xp.shape != X.shape) or np.any(Xm.shape != X.shape):
			raise ValueError('Xp or Xm has illegal shape')
		yp,ym = [f(Xp),f(Xm)] if hess is not None else [f(Xp)[0],f(Xm)[0]]
		hessN[:,i,i] = (yp - 2*fc + ym) / eps2

		for j in range(i+1,D):
			# Add steps to test points
			Tj = get_step_array(j)
			Xpp, Xpm, Xmp, Xmm = Xp+Tj, Xp-Tj, Xm+Tj, Xm-Tj
			ypp,ypm,ymp,ymm = [f(Xpp),f(Xpm),f(Xmp),f(Xmm)] if hess is not None \
						else [f(Xpp)[0],f(Xpm)[0],f(Xmp)[0],f(Xmm)[0]]
			hessN[:,i,j] = (ypp - ypm - ymp + ymm) / (4*eps2)
			hessN[:,j,i] = hessN[:,i,j]

	# Compute difference in gradients
	gradDiff = np.sqrt((hessA-hessN)**2/(hessA+hessN)**2)

	if not flatten: return gradDiff,hessA,hessN
	else: return gradDiff,np.c_[hessA.flatten(),hessN.flatten()]





def example ():
	"""
	Example of how to use checkgrad to check gradient.
	"""
	# 2-dimensional
	def f (X):
		func = X[:,0]**2 - X[:,1]
		grad = np.column_stack((2*X[:,0],-np.ones(X.shape[0])))
		return func, grad
	X = np.array([[1,2],[3,4],[5,6]])
	d,dy,dh = check_gradient(f,X)
	
	# 1-dimensional
	def f2 (X):
		return 3*X**2, 6*X
	X = np.array([[1],[2],[3]])
	d,dydh = check_gradient(f2,X,flatten=True)

