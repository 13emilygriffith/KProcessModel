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
		elements array. Bad data are filled in as zeros
	allivars: numpy array shape(N, M)
		Array of inverse variance on [X/H] abundance values with columns
		corresponding to the elmeents array. Bad data are filled in as zeros
	M: int
		Number of elements
	N: int
		Number of stars

	"""

	def __init__(self, elements, alldata, allivars):
		
		if isinstance(elements, np.ndarray): pass
		else:
			raise TypeError("Attribute 'elements' must be a numpy array."
				"Got: %s" % (type(elements)))
		
		if len(elements)>0: pass
		else:
			raise TypeError("The array 'elements' must contain at least one"
				"item. Got empty array")
			
		if isinstance(alldata, np.ndarray): pass
		else:
			raise TypeError("Attribute 'alldata' must be a numpy array."
				"Got: %s" % (type(alldata)))
			
		if len(alldata)>0: pass
		else:
			raise TypeError("The array 'alldata' must contain at least one"
				"item. Got empty array")
			
		if isinstance(allivars, np.ndarray):
			pass
		else:
			raise TypeError("Attribute 'allivars' must be a numpy array."
				"Got: %s" % (type(allivars)))
			
		if len(allivars)>0: pass
		else:
			raise TypeError("The array 'allivars' must contain at least one"
				"item. Got empty array")
			
		if np.shape(alldata)==np.shape(allivars): pass
		else:
			raise TypeError("The arrays 'alldata' and 'allivars' must be the"
				"same shape. Got shapes %s and %s" 
						   % (np.shape(alldata), np.shape(allivars)))
			
		if np.shape(alldata)[1] == len(elements): pass
		else:
			raise TypeError("The number of columns in 'alldata' must match the"
				"number of elements. Got lengths %s and %s" 
							% (np.shape(alldata)[1], len(elements)))
		
		if np.shape(allivars)[1] == len(elements): pass
		else:
			raise TypeError("The number of columns in 'allivars' must match"
				"the number of elements. Got lengths %s and %s" 
							% (np.shape(alldata)[1], len(elements)))
			
		if np.isfinite(alldata).all(): pass
		else:
			raise TypeError("All items in 'alldata' must be finite numbers.")
			
		if np.isfinite(allivars).all(): pass
		else:
			raise TypeError("All items in 'allivars' must be finite numbers.")
			
		if np.array([isinstance(val, str) for val in elements]).any(): pass
		else:
			raise TypeError("All items in 'elements' must be strings.")
		
		
		self._elements = elements
		self._alldata = alldata
		self._allivars = allivars
		self._allivars_orig = allivars
		self._M = len(self._elements)
		self._N = len(self._alldata)


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
	
	# What is the point of doing properties? Why not just remove _ above?
	@property
	def elements(self):
		return self._elements    
	
	@property
	def alldata(self):
		return self._alldata
		
	@property
	def allivars(self):
		return self._allivars

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


	@allivars.setter
	def allivars(self, new_ivars):
		"""
		Class function that updates the ivars and sqrt_ivars params with new 
		values. Used to inflate the ivars.
		"""

		if isinstance(new_ivars, (np.ndarray, jnp.ndarray)): pass
		else:
			raise TypeError("Attribute 'allivars' must be an array."
				"Got %s" % (type(new_ivars)))
		if np.shape(new_ivars) != np.shape(self.allivars):
			raise TypeError("Attribute 'allivars' must have shape %f."
				"Got %f" % (np.shape(self.allivars)), np.shape(new_ivars))

		if np.isfinite(new_ivars).all(): pass
		else:
			raise TypeError("All items in 'allivars' must be finite numbers")

		self._allivars = new_ivars
		self._sqrt_allivars = np.sqrt(new_ivars)


class fixed_params:
	"""
	Parameters needed in KPM fitting routine

	Parameters
	----------
	K: int
		Number of processes to fit. Default is 2
	processes_all: numpy array shape(4)
		Names of all processes possible to fit. 
		Default is ['CC','Ia','third','fourth']
	processes: numpy array shape(K)
		Names of all processes desired in current fit. 
	Lambda_a: int
		Regularization strength on fixed parameters. 
		Default is 1.e6
	Lambda_c: int
		Regularization strength on not fixed q values. 
		Default is 1.e3
	Lambda_d: int
		Regularisation strength on A values for processes k>2. 
		Default is 1.e3
	CC_elem: str
		Element assumed to be purely produced by CC process.
		Default is Mg
	Ia_element: str
		Element assumed to have fixed produce in both CC and Ia process. 
		Default is Fe
	id_CC: int
		index of CC_element in AbundData.elements array
	id_Ia: int
		index of Ia_element in AbundData.elements array
	A_list: array
		list of element names used in the A step
	I: array
		same lenfth as data.elements, list of true/false if elem is in A_list
	q_CC_Fe: int
		Value of q_CC for Ia_element at Z=0. 
		Default is 0.4
	xs: numpy array shape(N)
		Metallicity values for each item. Default is [Mg/H]
	J: int
		Number of parameters in lnq_par model (only odd used)
	xlim: numpy array shape(2)
		min and max of xs
		**** what we care about

	Class Methods
	-------------
		
	
	TO DO 
	- values of lambdas based on size of sample?
	- dictionary of elements and ids for each process?
	"""
	
	def __init__(self, data, A_list, K=2, Lambda_a=1.e6, Lambda_c=1.e3,
				 Lambda_d=1.e3, CC_elem='Mg', Ia_elem='Fe', 
				 q_CC_Fe=0.4, J=9):
		
		if isinstance(K, int): pass
		else:
			raise TypeError("Attribute 'K' must be an int. Got: %s" 
				% (type(K)))
		if K>4: 
			raise ValueError("Atrribute 'K' cannot exceed 4. Got %d" % (K))
			
		if isinstance(Lambda_a, float): pass
		else:
			raise TypeError("Attribute 'Lambda_a' must be a float. Got: %s" 
				% (type(Lambda_a)))
			
		if isinstance(Lambda_c, float): pass
		else:
			raise TypeError("Attribute 'Lambda_c' must be a float. Got: %s" 
				% (type(Lambda_c)))
			
		if isinstance(Lambda_d, float): pass
		else:
			raise TypeError("Attribute 'Lambda_d' must be a float. Got: %s" 
				% (type(Lambda_d)))
			
		if isinstance(CC_elem, str): pass
		else:
			raise TypeError("Attribute 'CC_elem' must be an str. Got: %s" 
				% (type(CC_elem)))
		if CC_elem not in data.elements:
			raise ValueError("Attribute 'CC_elem' %s is not in"
				"AbundData.elements" % (CC_elem))
			
		if isinstance(Ia_elem, str): pass
		else:
			raise TypeError("Attribute 'Ia_elem' must be an str. Got: %s" 
				% (type(Ia_elem)))
		if Ia_elem not in data.elements:
			raise ValueError("Attribute 'Ia_elem' %s is not in \
				AbundData.elements" % (CC_elem))

		if isinstance(A_list, np.ndarray): pass
		else:
			raise TypeError("Attribute 'elements' must be a numpy array."
				"Got: %s" % (type(A_list)))
		
		if len(A_list)>0: pass
		else:
			raise TypeError("The array 'A_list' must contain at least one"
				"item. Got empty array")
		## Need to check that all are in data.elements!!
		## Need to check that CC_elem and Ia_elem are in A_list!!
			
		if isinstance(q_CC_Fe, float): pass
		elif isinstance(q_CC_Fe, int): q_CC_Fe = float(q_CC_Fe)
		else:
			raise TypeError("Attribute 'q_CC_Fe' must be a float. Got: %s" 
				% (type(q_CC_Fe)))
		if (q_CC_Fe > 1.) | (q_CC_Fe < 0.): 
			raise ValueError("Atrribute 'q_CC_Fe' must lie between 0 and 1."
				"Got %.2f" % (q_CC_Fe))
			
		if isinstance(J, int): pass
		else:
			raise TypeError("Attribute 'J' must be an int. Got: %s" 
				% (type(J)))
		if J%2==0: raise warnings.warn("The number of parameters in the q \
			model must be odd. Got J= %f, using %f" % (J, J-1))
		
		
		self._K = K
		self._processes_all = np.array(['CC','Ia','third','fourth'])
		self._Lambda_a = Lambda_a
		self._Lambda_c = Lambda_d
		self._Lambda_d = Lambda_c
		self._CC_elem = CC_elem
		self._Ia_elem = Ia_elem
		self._q_CC_Fe = q_CC_Fe
		self._J = J
		
		self._id_CC = np.where(data.elements==CC_elem)[0][0]
		self._id_Ia = np.where(data.elements==Ia_elem)[0][0]
		self._elements = data.elements

		self._processes = self._processes_all[:K]

		sqrt_Lambda = jnp.ones(self._K) * jnp.sqrt(self._Lambda_d)
		sqrt_Lambda = jnp.where(self._processes == "CC", 0., sqrt_Lambda)
		sqrt_Lambda = jnp.where(self._processes == "Ia", 0., sqrt_Lambda)
		self._sqrt_Lambda_A = sqrt_Lambda

		self._xs = data.alldata[:,self._id_CC]
		self._alldata =data.alldata # THIS SEEMS BAD

		self._xlim =  np.percentile(self._xs, [1,99]) #hack
		self._L = self._xlim[0] - self._xlim[1]

		self._A_list = A_list
		self._I = [el in A_list for el in data.elements]
		
	def __repr__(self):
		attrs = {
			"K":				self._K,
			"Processes":		self._processes,
			"CC element":		self._CC_elem,
			"Ia element":		self._Ia_elem,
			"q_CC_Fe":			self._q_CC_Fe,
			"J":				self._J,
			"Lambda a":			self._Lambda_a,
			"Lambda c":			self._Lambda_c,
			"Lambda_d":			self._Lambda_d,
			"xlim":				self._xlim
		}

		rep = "kpm.fixed_params{\n"
		for i in attrs.keys():
			rep += "    %s " % (i)
			for j in range(20 - len(i)):
				rep += '-'
			rep += " > %s\n" % (str(attrs[i]))
		rep += '}'
		return rep

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
		self._processes = self._processes_all[:self._K]
	
	@property
	def processes_all(self):
		return self._processes_all
	
	@property
	def processes(self):
		return self._processes
	
	@property
	def Lambda_a(self):
		return self._Lambda_a
	
	@Lambda_a.setter
	def Lambda_a(self, value):
		if isinstance(value, float): pass
		else:
			raise TypeError("Attribute 'Lambda_a' must be a float. Got: %s" 
				% (type(value)))
			
		self._Lambda_a = value
		self._sqrt_Lambda_A = np.sqrt(value)
	
	@property
	def Lambda_c(self):
		return self._Lambda_c
	
	@Lambda_c.setter
	def Lambda_c(self, value):
		if isinstance(value, float): pass
		else:
			raise TypeError("Attribute 'Lambda_c' must be a float. Got: %s" 
				% (type(value)))
			
		self._Lambda_c = value 
	
	@property
	def Lambda_d(self):
		return self._Lambda_d
	
	@Lambda_d.setter
	def Lambda_d(self, value):
		if isinstance(value, float): pass
		else:
			raise TypeError("Attribute 'Lambda_d' must be a float. Got: %s" 
				% (type(value)))
			
		self._Lambda_d = value 

	@property
	def sqrt_Lambda_A(self):
		return self._sqrt_Lambda_A

	
	@property
	def CC_elem(self):
		return self._CC_elem
	
	@CC_elem.setter
	def CC_elem(self, value):
		if isinstance(value, str): pass
		else:
			raise TypeError("Attribute 'CC_elem' must be an str. Got: %s" 
				% (type(value)))
		if value not in self._elements:
			raise ValueError("Attribute 'CC_elem' %s is not in"
				"abund_data.elements" % (value))
		self._CC_elem = value
		self._id_CC = np.where(self._elements==value)[0][0]
		self._xs = self._alldata[:,self._id_CC]
	
	@property
	def Ia_elem(self):
		return self._Ia_elem
	
	@Ia_elem.setter
	def Ia_elem(self, value):
		if isinstance(value, str): pass
		else:
			raise TypeError("Attribute 'Ia_elem' must be an str. Got: %s" 
				% (type(value)))
		if value not in self._elements:
			raise ValueError("Attribute 'Ia_elem' %s is not in"
				"abund_data.elements" % (value))
		self._Ia_elem = value
		self._id_Ia = np.where(self._elements==value)[0][0]
	
	@property
	def q_CC_Fe(self):
		return self._q_CC_Fe
	
	@q_CC_Fe.setter
	def q_CC_Fe(self, value):
		if isinstance(value, float): pass
		elif isinstance(value, int): value = float(value)
		else:
			raise TypeError("Attribute 'q_CC_Fe' must be a float. Got: %s" 
				% (type(value)))
		if (value > 1.) | (value < 0.): 
			raise ValueError("Atrribute 'q_CC_Fe' must lie between 0 and 1."
				"Got %.2f" % (value))
		self._q_CC_Fe = value
		
	@property
	def id_CC(self):
		return self._id_CC
	
	@property
	def id_Ia(self):
		return self._id_Ia

	@property
	def xs(self):
		return self._xs

	@property
	def J(self):
		return self._J
	
	@J.setter
	def J(self, value):
		if isinstance(value, int): pass
		else:
			raise TypeError("Attribute 'J' must be an int. Got: %s" 
				% (type(value)))
		if J%2==0: raise warnings.warn("The number of parameters in the q \
			model must be odd. Got J= %f, using %f" % (J, J-1))
		self._J = value


	@property
	def xlim(self):
		return self._xlim

	@xlim.setter
	def xlim(self, value):
		print('reseting xlims!!!!!!!')

		if isinstance(value, np.ndarray): pass
		else: 
			raise TypeError("Attribute 'xlim' must be an array. Got: %s" 
				% (type(value)))
		if len(value)==2: pass
		else: 
			raise TypeError("Attribute 'xlim' must have length 2. Got: %f" 
				% (len(value)))

		#Check that xlim overlaps with xs!!!!
		self._xlim = value
		self._L = value[0] - value[1]

	@property
	def L(self):
		return self._L

	@property
	def A_list(self):
		return self._A_list

	@A_list.setter
	def A_list(self, value):
		#ADD WARNING FLAGS
		I = [el in value for el in data.elements]
		self._A_list = value 
		self._I = I

	@property
	def I(self):
		return self._I
	

class fit_params:
	"""
	Add Text

	To Do: Check that lnq and lnA arrays are the right size
	Verify that this throws a flag if the user trys to set something of the wrong shape
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
		self._lnq_pars = value

	@lnAs.setter
	def lnAs(self, value):
		self._lnAs = value








