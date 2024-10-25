import numpy as np
import warnings
import jax.numpy as jnp

__all__ = ["abund_data", "fixed_params", "fit_params"]

class abund_data:
	"""
	Stellar abundances and inverse variances for a dataset

	Parameters
	----------
	elements: numpy array shape(M)
		Array of included elemental abundances. Element names must be strings
	alldata: numpy array shape(N, M)
		Array of [X/H] abundance values with columns corresponding to the 
		elements array. Bad data are filled in as zeros.
	allivars: numpy array shape(N, M)
		Array of inverse variance on [X/H] abundance values with columns
		corresponding to the elmeents array. Bad data are filled in as zeros.
		Updated with inflated values through optimization routine
	sqrt_allivars; numpy array shape(N, M)
		Array of squareroot of inverse variance on [X/H] abundances values
	allivars_orig: 
		Array of inverse variance on [X/H] abundance values with columns
		corresponding to the elemnet array. Bad data are filled in as zeros.
		Preserves original ivars while 'allivars' array is updated with 
		inflated values.
	M: int
		Number of elements
	N: int
		Number of stars


	TO DO
	-----
	- Check if alldata and allivar elements are floats
	"""

	def __init__(self, elements, alldata, allivars):

		self.elements = elements
		self.alldata = alldata
		self.allivars = allivars
		self._allivars_orig = allivars
			
	@property
	def elements(self):
		return self._elements   

	@elements.setter
	def elements(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'elements' must be a numpy array."
				"Got: %s" % (type(value))) 
		if len(value)>0: pass
		else:
			raise TypeError("The array 'elements' must contain at least one"
				"item. Got empty array")
		if np.array([isinstance(val, str) for val in value]).any(): pass
		else:
			raise TypeError("All items in 'elements' must be strings.")
		self._elements = value
		self._M = len(self._elements)
	
	@property
	def alldata(self):
		return self._alldata

	@alldata.setter
	def alldata(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'alldata' must be a numpy array."
				"Got: %s" % (type(value)))
		if len(value)>0: pass
		else:
			raise TypeError("The array 'alldata' must contain at least one"
				"item. Got empty array")
		# Is there an easy way to check if all elements of alldata are floats?
		if np.isfinite(value).all(): pass
		else:
			raise TypeError("All items in 'alldata' must be finite numbers")
		if np.shape(value)[1] == len(self._elements): pass
		else:
			raise TypeError("The number of columns in 'alldata' must match the"
				"number of elements. Got lengths %s and %s" 
				% (np.shape(value)[1], len(self._elements)))
		self._alldata = value
		self._N = len(self._alldata)

	@property
	def allivars(self):
		return self._allivars

	@allivars.setter
	def allivars(self, new_ivars):
		if isinstance(new_ivars, (np.ndarray, jnp.ndarray)): pass
		else:
			raise TypeError("Attribute 'allivars' must be an array."
				"Got %s" % (type(new_ivars)))
		if np.shape(new_ivars) != np.shape(self._alldata):
			raise TypeError("Attribute 'allivars' must have the same shape as" 
				" 'alldata' %f Got %f" % (np.shape(self._alldata)), 
				np.shape(new_ivars))
		if np.isfinite(new_ivars).all(): pass
		else:
			raise TypeError("All items in 'allivars' must be finite numbers")

		self._allivars = new_ivars
		self._sqrt_allivars = np.sqrt(new_ivars)

	@property
	def allivars_orig(self):
		return self._allivars_orig

	@property
	def sqrt_allivars(self):
		return np.sqrt(self._allivars)
	
	@property 
	def M(self):
		return self._M
	
	@property 
	def N(self):
		return self._N

	def __repr__(self):
		attrs = {
			"Elements":               self._elements,
			"Number of elements":     self._M,
			"Number of stars":        self._N
		}

		rep = "kpm.abund_data{\n"
		for i in attrs.keys():
			rep += "    %s " % (i)
			for j in range(20 - len(i)):
				rep += '-'
			rep += " > %s\n" % (str(attrs[i]))
		rep += '}'
		return rep


class fixed_params:
	"""
	Parameters needed in KPM fitting routine

	Parameters
	----------
	K: int
		Number of processes to fit. Cannot be greater than 4
		Default is 2
	J: int
		Number of parameters in lnq_par model. Can only be odd. If given an
		even number, will use J-1
		Default is 9
	elements: numpy array shape(M)
		List of elements to fit with q_step. Should be identical to list in 
		'abund_data' class.
	A_list: numpy array
		Array of element names used in the A step. Must be subset of 'elements'
		Default is ['Mg', 'Fe']
	I: numpy array shape(M)
		Array of True/False if each element in 'elements' is in 'A_list'
	proc_elems: numpy array shape(K)
		Array of element names that each process is fixed to. Must be length K
		Default is ['Mg', 'Fe']
	proc_ids: numpy array shape(K)
		Array of the index number in 'elements' for e in 'proc_elem'
	xs: numpy array shape(N)
		Metallicity values for each item, taken as the [X/H] for the first
		element in proc_elems
		Default is [Mg/H]
	xlim: numpy array shape(2)
		Array with the min and max of xs. This is the range that the q vectors
		are fit over
		Default is 1st and 99th percentile of xs
	L: float
		The length of the metallicity space being fit. xlim[1] - xlim[0]
	q_fixed: numpy array shape(K, K)
		Array with the fixed values of the q_vectors for each fixed element in 
		proc_elems. Index corresponds to the process number
		Default is [[1.,0.], [0.4,0.6]], such that Mg is purely produced by the 
		first process and Fe is dominantly produced by the second process
	Lambda_As: numpy array shape(K)
		Array of regularization strengths on A values. One regularization
		per process. 
		Default is [1.e3, 1.e3]
	Lambda_qs: numpy array shape(2)
		Array of regularization strength on q values. The zeroth element is the
		regularization strength on the fixed q values. The first element is the
		regularization strength on the free q values.
		Default is [1.e6, 1.e3]
	sqrt_Lambda_As: numpy array shape(K)
		Array of squareroot of the regularisation strengths on A values.
	ln_noise: float
		Log of hacky amplitude of noise used to improve the optimization
		Default is -4.0
	Delta: float
		Value of dilution coefficient to be applied to all stars
		Default is 0.0

	Class Methods
	-------------
		
	
	TO DO 
	-----
	- values of lambdas based on size of sample?
	- Add notes on default values to the header
	- K has a value error that it can't be greater than 4. This should be 
		relaxed?
	- create a class method to update proc_elems that takes data class as input.
	- create better _repr_ function
	- right now many things have some shape connected to K. Need warning if user
		resets K but not other values
	"""
	
	def __init__(self, data, K=2, J=9,
				 A_list=np.array(['Mg','Fe']),
				 proc_elems=np.array(['Mg','Fe']), 
				 q_fixed=np.array([[1.,0.],[0.4,0.6]]),
				 Lambda_qs=np.array([1.e6, 1.e3]),
				 Lambda_As=np.array([0, 0]),
				 ln_noise=-4.0, Delta=0.0):

		self.K = K
		self.J = J
		self._elements = data.elements
		self.A_list = A_list

		if isinstance(proc_elems, np.ndarray): pass
		else:
			raise TypeError("Attribute 'proc_elems' must be an array. Got: %s" 
				% (type(proc_elems)))
		for e in proc_elems:
			if isinstance(e, str): pass
			else:
				raise TypeError("Elements of 'proc_elems' must be strings." 
					"Got %s" % (type(e)))
			if e not in self._elements:
				raise ValueError("Element of 'proc_elems' %s is not in"
					"initialized list of elements" % (e))
		self._proc_elems = proc_elems
		self._proc_ids = np.array([np.where(self._elements==e)[0][0] 
								   for e in proc_elems])
		self._xs = data.alldata[:,self._proc_ids[0]]
		self._xlim =  np.percentile(self._xs, [1,99]) #hack
		self._L = self._xlim[0] - self._xlim[1]

		self.q_fixed = q_fixed
		
		self.Lambda_qs = Lambda_qs
		self.Lambda_As = Lambda_As

		self._ln_noise = ln_noise

		self._Delta = Delta

	@property
	def K(self):
		return self._K

	@K.setter
	def K(self, value):
		if isinstance(value, int): pass
		else: raise TypeError("Attribute 'K' must be an int. Got: %s" 
			% (type(value)))
		if value>4: 
			raise ValueError("Atrribute 'K' cannot exceed 4. Got %d" % (value))
		self._K = value

	@property
	def J(self):
		return self._J
	
	@J.setter
	def J(self, value):
		if isinstance(value, int): pass
		else:
			raise TypeError("Attribute 'J' must be an int. Got: %s" 
				% (type(value)))
		if value%2==0: 
			raise warnings.warn("The number of parameters in the q \
			model must be odd. Got J= %f, using %f" % (value, value-1))
			self._J = value-1
		else: self._J = value

	@property
	def elements(self):
		return self._elements

	@property
	def A_list(self):
		return self._A_list

	@A_list.setter
	def A_list(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'elements' must be None type or a" 
				"numpy array. Got: %s" % (type(value)))
		if len(value)>0: pass
		else:
			raise TypeError("The array 'A_list' must contain at least one"
				"item. Got empty array")
		for e in value:
			if isinstance(e, str): pass
			else:
				raise TypeError("Elements of 'A_list' must be strings."
					"Got %s" % (type(e)))
			if e not in self._elements:
				raise ValueError("Element of 'proc_elem' %s is not in"
					"initialized list of elements" % (e))	
		self._A_list = value		
		I = [el in value for el in self._elements]
		self._I = I

	@property
	def I(self):
		return self._I

	@property
	def proc_elems(self):
		return self._proc_elems
	
	@proc_elems.setter
	def proc_elems(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'proc_elems' must be an array. Got: %s" 
				% (type(value)))
		if len(value) != self._K:
			raise ValueError("Length of 'proc_elems' must be equal to K."
				"Got %s" % (len(value)))
		for e in value:
			if isinstance(e, str): pass
			else:
				raise TypeError("Elements of 'proc_elems' must be strings." 
					"Got %s" % (type(e)))
			if e not in self._A_list:
				raise ValueError("Element of 'proc_elems' %s is not in"
					"A_list" % (e))
		if value[0] != self._proc_elems[0]:
			raise ValueError("Cannot reset the first process element. Please \
				re-initialize 'fixed_params'.")
		self._proc_elems = value
		self._proc_ids = np.array([np.where(self._elements==e)[0][0] for e in value])

	@property
	def proc_ids(self):
		return self._proc_ids

	@property
	def xs(self):
		return self._xs

	@property
	def xlim(self):
		return self._xlim

	@xlim.setter
	def xlim(self, value):
		if isinstance(value, np.ndarray): pass
		else: 
			raise TypeError("Attribute 'xlim' must be an array. Got: %s" 
				% (type(value)))
		if len(value)==2: pass
		else: 
			raise TypeError("Attribute 'xlim' must have length 2. Got: %f" 
				% (len(value)))
		if value[0] < value[1]: pass
		else:
			raise ValueError("First element of 'xlim' must be less than the \
				second element.")
		# if (value[0] > np.nanmax(self._xs)) or (value[1] < np.nanmin(self._xs)):
		# 	raise ValueError("Attribute 'xlim' must overlap with 'xs' whose \
		# 		minimum is %s and maximum is %s" % (np.nanmin(self._xs,
		# 			np.nanmax(self._xs))))

		self._xlim = value
		self._L = value[0] - value[1]

	@property
	def L(self):
		return self._L

	@property
	def q_fixed(self):
		return self._q_fixed

	@q_fixed.setter
	def q_fixed(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'q_fixed' must be a numpy array."
				"Got: %s" % (type(value)))
		if np.shape(value) != (self._K,self._K): 
			raise ValueError("Shape of 'q_fixed' must be ('K', 'K'). Got (%s, %s)" 
				% (np.shape(value)[0], np.shape(value)[1]))
		for q in value.flatten():
			if isinstance(q, float):
				if q<0: 
					raise ValueError("Elements of 'q_fixed' must be non-"
						"negative. Got %s" % (q))
				elif q>2:
					raise warnings.warn("We recomend fixed q values between"
						"0 and 1. Got %s" % q)
			elif (q is None): pass
			else:
				raise ValueError("Elements of 'q_fixed' must be floats or None."
					"Got %s" % (type(q)))
		self._q_fixed = value

	@property
	def Lambda_As(self):
		return self._Lambda_As

	@Lambda_As.setter
	def Lambda_As(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'Lambda_As' must be a numpy array."
				"Got: %s" % (type(value)))
		if len(value) != self._K:
			raise ValueError("Length of 'Lambda_As must be equal to K.\
				Got %s" % (len(value)))
		for L in value:
			if isinstance(L, float): pass
			else: 
				raise TypeError("Elements of 'Lambda_As' must be floats."
				"Got: %s" % (type(L)))
		self._Lambda_As = value
		self._sqrt_Lambda_As = np.sqrt(value)

	@property
	def sqrt_Lambda_As(self):
		return self._sqrt_Lambda_As

	@property
	def Lambda_qs(self):
		return self._Lambda_qs
	
	@Lambda_qs.setter
	def Lambda_qs(self, value):
		if isinstance(value, np.ndarray): pass
		else:
			raise TypeError("Attribute 'Lambda_qs' must be a numpy array."
				"Got: %s" % (type(value)))
		if len(value) != 2:
			raise ValueError("Length of 'Lambda_qs must be 2. Got %s"
				% (len(value)))
		for L in value:
			if isinstance(L, float): pass
			else: 
				raise TypeError("Elements of 'Lambda_qs' must be floats."
				"Got: %s" % (type(L)))
		self._Lambda_qs = value

	@property
	def ln_noise(self):
		return self._ln_noise

	@ln_noise.setter
	def ln_noise(self, value):
		self._ln_noise = value

	@property
	def Delta(self):
		return self._Delta

	@Delta.setter
	def Delta(self, value):
		if isinstance(value, float): pass
		else: raise TypeError("Attribute 'Delta' must be a float. Got: %s" 
			% (type(value)))
		# if np.abs(value)>2: 
		# 	# Should warn for large value
		# 	#raise ValueError("You have chosen a very large Delta. Got %d" % (value))
		self._Delta = value
		
	def __repr__(self):
		attrs = {
			"K":				self._K,
			"A list":			self._A_list,
			"Proc elements":	self._proc_elems,
			"fixed qs":			self._q_fixed,
			"J":				self._J,
			"Lambda As":		self._Lambda_As,
			"Lambda qs":		self._Lambda_qs,
			"xlim":				self._xlim,
			"Delta":			self._Delta
		}

		rep = "kpm.fixed_params{\n"
		for i in attrs.keys():
			rep += "    %s " % (i)
			for j in range(20 - len(i)):
				rep += '-'
			rep += " > %s\n" % (str(attrs[i]))
		rep += '}'
		return rep
	

class fit_params:
	"""
	Parameters fit in KPM routine

	Parameters
	----------
	lnq_pars: numpy array shape(K, J, M)
		Array of log of coefficients for the q process vectors

	lnAs: numpy array shape(K, N)
		Array of log of process amplitudes for each star
	"""

	def __init__(self, data, fixed):

		lnq_pars = np.zeros((fixed.K, fixed.J, data.M))
		lnAs = np.zeros((fixed.K, data.N))

		self._lnq_pars = lnq_pars
		self._lnAs = lnAs 

	@property
	def lnq_pars(self):
		return self._lnq_pars

	@property
	def lnAs(self):
		return self._lnAs

	@lnq_pars.setter
	def lnq_pars(self, value):
		if np.shape(value) == np.shape(self._lnq_pars): pass
		else: 
			raise ValueError("Attribute 'lnq_pars' must be shape %s. Got %s"
				% (np.shape(self._lnq_pars), np.shape(value)))
		self._lnq_pars = value

	@lnAs.setter
	def lnAs(self, value):
		if np.shape(value) == np.shape(self._lnAs): pass
		else: 
			raise ValueError("Attribute 'lnAs' must be shape %s. Got %s"
				% (np.shape(self._lnAs), np.shape(value)))
		self._lnAs = value








