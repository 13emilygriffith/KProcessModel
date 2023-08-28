import numpy as np
import warnings
import jax.numpy as jnp

__all__ = ["AbundData", "FixedParams", "FitParams"]

class AbundData:
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


class FixedParams:
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
    q_CC_Fe: int
        Value of q_CC for Ia_element at Z=0. 
        Default is 0.4
    dq_CC_Fe_dZ: int
        Value of slope in q_CC for Ia_element with respect to Z. 
        Default is 0.0
    xs: numpy array shape(N)
    	Metallicity values for each item. Default is [Mg/H]
    Nknot: int
    	Number of knots in the spine for the process vectors
    knot_xs: numpy array shape(K, Nknot)
    	Array of knots for each process. Default is values at 'Nknot' linearly
    	spaced percintiles of 'xs'

    Class Methods
    -------------
    update_knot_xs: function
    	User can suply their own array of knots
        
    
    TO DO 
    - values of lambdas based on size of sample?
    - dictionary of elements and ids for each process?
    
    """
    
    def __init__(self, AbundData, K=2, Lambda_a=1.e6, Lambda_c=1.e3,
                 Lambda_d=1.e3, CC_elem='Mg', Ia_elem='Fe', 
                 q_CC_Fe=0.4, dq_CC_Fe_dZ=0.0, Nknot=10):
        
        if isinstance(K, int): pass
        else:
            raise TypeError("Attribute 'K' must be an int. Got: %s" 
            	% (type(K)))
        if K>4: 
            raise TypeError("Atrribute 'K' cannot exceed 4. Got %d" % (K))
            
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
        if CC_elem not in AbundData.elements:
            raise TypeError("Attribute 'CC_elem' %s is not in"
            	"AbundData.elements" % (CC_elem))
            
        if isinstance(Ia_elem, str): pass
        else:
            raise TypeError("Attribute 'Ia_elem' must be an str. Got: %s" 
            	% (type(Ia_elem)))
        if Ia_elem not in AbundData.elements:
            raise TypeError("Attribute 'Ia_elem' %s is not in \
            	AbundData.elements" % (CC_elem))
            
        if isinstance(q_CC_Fe, float): pass
        elif isinstance(q_CC_Fe, int): q_CC_Fe = float(q_CC_Fe)
        else:
            raise TypeError("Attribute 'q_CC_Fe' must be a float. Got: %s" 
            	% (type(q_CC_Fe)))
        if (q_CC_Fe > 1.) | (q_CC_Fe < 0.): 
            raise TypeError("Atrribute 'q_CC_Fe' must lie between 0 and 1."
            	"Got %.2f" % (q_CC_Fe))
            
        if isinstance(dq_CC_Fe_dZ, float): pass
        elif isinstance(dq_CC_Fe_dZ, int): dq_CC_Fe_dZ = float(dq_CC_Fe_dZ)
        else:
            raise TypeError("Attribute 'q_CC_Fe' must be a float. Got: %s" 
            	% (type(q_CC_Fe)))

        if isinstance(Nknot, int): pass
        else:
            raise TypeError("Attribute 'Nknot' must be an int. Got: %s" 
            	% (type(Nknot)))
        
        
        self._K = K
        self._processes_all = np.array(['CC','Ia','third','fourth'])
        self._Lambda_a = Lambda_a
        self._Lambda_c = Lambda_d
        self._Lambda_d = Lambda_c
        self._CC_elem = CC_elem
        self._Ia_elem = Ia_elem
        self._q_CC_Fe = q_CC_Fe
        self._dq_CC_Fe_dZ = dq_CC_Fe_dZ
        self._Nknot = Nknot
        
        self._id_CC = np.where(AbundData.elements==CC_elem)[0][0]
        self._id_Ia = np.where(AbundData.elements==Ia_elem)[0][0]

        self._processes = self._processes_all[:K]

        sqrt_Lambda = jnp.ones(self._K) * jnp.sqrt(self._Lambda_d)
        sqrt_Lambda = jnp.where(self._processes == "CC", 0., sqrt_Lambda)
        sqrt_Lambda = jnp.where(self._processes == "Ia", 0., sqrt_Lambda)
        self._sqrt_Lambda_A = sqrt_Lambda

        self._xs = AbundData.alldata[:,self._id_CC]

        self._knot_ps = np.percentile(self._xs, np.linspace(0,100,Nknot))
        self._knot_xs = np.ones([self._K, self._Nknot]) * self._knot_ps
        
    @property
    def K(self):
    	return self._K

    @K.setter
    #Note that if you change K the knots and processes auto reset 
    def K(self, value):
    	if isinstance(value, int): pass
    	else: raise TypeError("Attribute 'K' must be an int. Got: %s" 
    		% (type(value)))
    	if value>4: 
    		raise TypeError("Atrribute 'K' cannot exceed 4. Got %d" % (value))
    	self._K = value
    	self._processes = self._processes_all[:self._K]
    	self._knot_xs = np.ones([self._K, self._Nknot]) * self._knot_ps

    
    @property
    def processes_all(self):
        return self._processes_all
    
    @property
    def processes(self):
        return self._processes
    
    @property
    def Lambda_a(self):
        return self._Lambda_a
    
    @Lambda_A.setter
    def Lambda_A(self, value):
        self._Lambda_A = value 
    
    @property
    def Lambda_c(self):
        return self._Lambda_c
    
    @property
    def Lambda_d(self):
        return self._Lambda_d 

    @property
    def sqrt_Lambda_A(self):
    	return self._sqrt_Lambda_A
    
    @property
    def CC_elem(self):
        return self._CC_elem
    
    @property
    def Ia_elem(self):
        return self._Ia_elem
    
    @property
    def q_CC_Fe(self):
        return self._q_CC_Fe 
    
    @property
    def dq_CC_Fe_dZ(self):
        return self._dq_CC_Fe_dZ
        
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
    def Nknot(self):
    	return self._Nknot

    @property
    def knot_xs(self):
    	return self._knot_xs

    # @property
    # def knot_ps(self):
    # 	return self._knot_ps

    @knot_xs.setter
    def knot_xs(self, new_knots):

    	if isinstance(new_knots, np.ndarray): pass
    	else: 
    		raise TypeError("Attribute 'knot_xs' must be an array. Got: %s" 
    			% (type(new_knots)))

		# if np.shape(new_knots)[0] != self._K: 
		# 	raise TypeError("Attribute 'knot_xs' must have two rows. Got: %s" % (np.shape(new_knots)[0]))

        # Must be in sequential order
        # Must overlap with the range of the xs

    	self._knot_xs = new_knots
    	self._Nknot = np.shape(new_knots)[1]

class FitParams:
	"""
	Add Text

	To Do: Check that lnq and lnA arrays are the right size
	Verify that this throws a flag if the user trys to set something of the wrong shape
	"""

	def __init__(self, AbundData, FixedParams):

		lnqs = np.zeros((FixedParams.K, FixedParams.Nknot, AbundData.M))
		lnAs = np.zeros((FixedParams.K, AbundData.N))

		self._lnqs = lnqs
		self._lnAs = lnAs 

	@property
	def lnqs(self):
		return self._lnqs

	@property
	def lnAs(self):
		return self._lnAs

	@lnqs.setter
	def lnqs(self, value):
		self._lnqs = value

	@lnAs.setter
	def lnAs(self, value):
		self._lnAs = value








