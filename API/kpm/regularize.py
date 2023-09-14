import numpy as np 
import jax.numpy as jnp

from ._globals import *

__all__ = ["regularizations"]

class regularizations:

	def __init__(self, data, fixed, init_q=0.1):

		self.A = self.A(fixed)
		self.Q = self.Q(data, fixed, init_q)

	class A:
		"""
		Add Text

		sqrt_Lambda_A: jax array shape(K)
		Array of squareroot of regularizations for all processes

		"""

		def __init__(self, fixed):

			sqrt_Lambda = jnp.ones(fixed.K) * jnp.sqrt(fixed.Lambda_d)
			sqrt_Lambda = jnp.where(fixed.processes == "CC", 0., sqrt_Lambda)
			sqrt_Lambda = jnp.where(fixed.processes == "Ia", 0., sqrt_Lambda)
			self._sqrt_Lambda_A = sqrt_Lambda

		@property
		def sqrt_Lambda_A(self):
			return self._sqrt_Lambda_A

	class Q:

		"""
		Add Text

		TO DO: ADD For K>2
		"""

		def __init__(self, data, fixed, init_q = 0.1):

			q_array = np.zeros((fixed.K, fixed.Nknot, data.M))
			Lambdas = q_array + fixed.Lambda_c
			q0s = q_array + init_q
			fixed_q = q_array.astype(bool)

			# 1: Strongly require that q_X = 1 for CC process for CC_elem
			proc = fixed.processes == "CC"
			elem = data.elements == fixed.CC_elem
			Lambdas[proc, :, elem] = fixed.Lambda_a
			q0s[    proc, :, elem] = 1.0
			fixed_q[  proc, :, elem] = True

			# 2: Require that q_X = 0 for ALL except CC process for CC_elem
			proc = fixed.processes != "CC"
			elem = data.elements == fixed.CC_elem
			Lambdas[proc, :, elem] = fixed.Lambda_a
			q0s[    proc, :, elem] = 0.0
			fixed_q[  proc, :, elem] = True

			# 3: Require that q_X has some value / form for CC process for Ia_elem
			proc = fixed.processes == "CC"
			elem = data.elements == fixed.Ia_elem
			Lambdas[proc, :, elem] = fixed.Lambda_a
			q0s[    proc, :, elem] = fixed.q_CC_Fe + fixed.dq_CC_Fe_dZ * fixed.knot_xs[proc]
			fixed_q[  proc, :, elem] = True

			# 4: Strongly require that q_X sum to 1 for Ia_elem
			proc = fixed.processes == "Ia"
			elem = data.elements == fixed.Ia_elem
			Lambdas[proc, :, elem] = fixed.Lambda_a
			q0s[    proc, :, elem] = 1.0 - (fixed.q_CC_Fe + fixed.dq_CC_Fe_dZ * fixed.knot_xs[proc])
			fixed_q[  proc, :, elem] = True

			self._init_q = init_q
			self._Lambdas = Lambdas
			self._q0s = q0s
			self._lnq0s = np.log(np.clip(q0s, 1.e-7, None))
			self._fixed_q = fixed_q

		@property
		def init_q(self):
			return self._init_q

		@property
		def Lambdas(self):
			return self._Lambdas

		@property
		def q0s(self):
			return self._q0s

		@property
		def lnq0s(self):
			return self._lnq0s

		@property
		def fixed_q(self):
			return self._fixed_q
	
	

