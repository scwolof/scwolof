
import numpy as np 

"""
Method for checking that analytical gradients are correct,
using comparison with finite difference gradients.
(C) 2016 by Simon Olofsson
"""

def checkgrad (f, X, grad=None, eps = 0.0001):
	"""
	Check gradients of function.
	checkgrad (f, X, grad=None, eps = 0.0001)

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

	Outputs:
	d 		Difference in analytical and numerical
			gradient values: sqrt((dy-dh)**2/(dy+dh)**2)
	dy 		Analytical gradients
	dh 		Numerical gradients
	"""

	n,D = X.shape[0], 1 if X.ndim == 1 else X.shape[1]
	epsVec = eps*np.ones(n)
	zeroArray = np.zeros(n*D,dtype='float').reshape((n,D))

	# Analytical gradient
	if grad is None:
		_,gradA = f(X)
	else:
		gradA = grad(X)

	# Numerical gradients using finite differences
	gradN = zeroArray.copy()
	for k in range(0,D):
		# Step array
		if D == 1:
			testarray = epsVec.copy().reshape(X.shape)
		else:
			testarray = zeroArray.copy()
			testarray[:,k] = epsVec
		# Add steps to test points
		Xp = X + testarray
		Xm = X - testarray
		if np.any(Xp.shape != X.shape) or np.any(Xm.shape != X.shape):
			raise ValueError('Xp or Xm has illegal shape')
		# Evaluate function
		if grad is None:
			y1,_ = f(Xp)
			y2,_ = f(Xm)
		else:
			y1 = f(Xp)
			y2 = f(Xm)
		# Finite difference computation
		if D == 1:
			gradN = (y1-y2)/(2*eps)
		else:
			gradN[:,k] = (y1-y2)/(2*eps)

	# Compute difference in gradients
	gradDiff = np.sqrt((gradA-gradN)**2/(gradA+gradN)**2)

	return gradDiff,gradA,gradN




def example ():
	"""
	Example of how to use checkgrad.
	"""
	# 2-dimensional
	def f (X):
		func = X[:,0]**2 - X[:,1]
		grad = np.column_stack((2*X[:,0],-np.ones(len(X[:,1]))))
		return func, grad
	X = np.array([[1,2],[3,4],[5,6]])
	d,dy,dh = checkgrad.checkgrad(f,X)
	
	# 1-dimensional
	def f2 (X):
		return 3*X**2, 6*X
	X = np.array([1,2,3])
	d,dy,dh = checkgrad.checkgrad(f2,X)







