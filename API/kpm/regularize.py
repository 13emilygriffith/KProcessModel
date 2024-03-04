import numpy as np 
import jax.numpy as jnp

from ._globals import *

__all__ = ["regularizations"]

class regularizations:

	def __init__(self, data, fixed):

		self.A = self.A_c(fixed)
		self.Q = self.Q_c(data, fixed)

	class A_c:
		"""
		Add Text

		sqrt_Lambda_A: jax array shape(K)
		Array of squareroot of regularizations for all processes

		"""

		def __init__(self, fixed):
			self._sqrt_Lambda_A = fixed.sqrt_Lambda_As

		@property
		def sqrt_Lambda_A(self):
			return self._sqrt_Lambda_A

	class Q_c:

		"""
		Add Text

		TO DO: ADD For K>2
		# q0s array is confusing. First element is the value of a flat line
		  subsequent values are the coefficients of cos/sin to deviate away 
		"""

		def __init__(self, data, fixed):

			q_array = np.zeros((fixed.K, fixed.J, data.M))
			# initially set all regularizations to the free reg value
			Lambdas = q_array + fixed.Lambda_qs[1]
			q0s = q_array + 1.0
			fixed_q = q_array.astype(bool)

			for i, elem_id in enumerate(fixed.proc_ids):
				for proc in range(fixed._K):
					value = fixed.q_fixed[i, proc]
					if value == None: pass
					else:
						Lambdas[proc, :, elem_id] = fixed.Lambda_qs[0]
						q0s[    proc, :, elem_id] = 1.0
						q0s[    proc, 0, elem_id] = value
						fixed_q[  proc, :, elem_id] = True

			self._Lambdas = Lambdas
			self._q0s = q0s
			self._lnq_par0s = np.log(np.clip(q0s, 1.e-7, None))
			self._fixed_q = fixed_q

		@property
		def Lambdas(self):
			return self._Lambdas

		@property
		def q0s(self):
			return self._q0s

		@property
		def lnq_par0s(self):
			return self._lnq_par0s

		@property
		def fixed_q(self):
			return self._fixed_q
	
	

