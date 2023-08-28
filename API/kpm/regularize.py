import numpy as np 
import jax.numpy as jnp

from ._globals import *

__all__ = ["Regs"]

class Regs:

	def __init__(self, AbundData, FixedParams, init_q=0.1):

		self.A = self.A(FixedParams)
		self.Q = self.Q(AbundData, FixedParams, init_q)

	class A:
		"""
		Add Text

		sqrt_Lambda_A: jax array shape(K)
		Array of squareroot of regularizations for all processes

		"""

		def __init__(self, FixedParams):

			sqrt_Lambda = jnp.ones(FixedParams.K) * jnp.sqrt(FixedParams.Lambda_d)
			sqrt_Lambda = jnp.where(FixedParams.processes == "CC", 0., sqrt_Lambda)
			sqrt_Lambda = jnp.where(FixedParams.processes == "Ia", 0., sqrt_Lambda)
			self._sqrt_Lambda_A = sqrt_Lambda

		@property
		def sqrt_Lambda_A(self):
			return self._sqrt_Lambda_A

	class Q:

		"""
		Add Text

		TO DO: ADD For K>2
		"""

		def __init__(self, AbundData, FixedParams, init_q = 0.1):

			q_array = np.zeros((FixedParams.K, FixedParams.Nknot, AbundData.M))
			Lambdas = q_array + FixedParams.Lambda_c
			q0s = q_array + init_q
			fixed = q_array.astype(bool)

			# 1: Strongly require that q_X = 1 for CC process for CC_elem
			proc = FixedParams.processes == "CC"
			elem = AbundData.elements == FixedParams.CC_elem
			Lambdas[proc, :, elem] = FixedParams.Lambda_a
			q0s[    proc, :, elem] = 1.0
			fixed[  proc, :, elem] = True

			# 2: Require that q_X = 0 for ALL except CC process for CC_elem
			proc = FixedParams.processes != "CC"
			elem = AbundData.elements == FixedParams.CC_elem
			Lambdas[proc, :, elem] = FixedParams.Lambda_a
			q0s[    proc, :, elem] = 0.0
			fixed[  proc, :, elem] = True

			# 3: Require that q_X has some value / form for CC process for Ia_elem
			proc = FixedParams.processes == "CC"
			elem = AbundData.elements == FixedParams.Ia_elem
			Lambdas[proc, :, elem] = FixedParams.Lambda_a
			q0s[    proc, :, elem] = FixedParams.q_CC_Fe + FixedParams.dq_CC_Fe_dZ * FixedParams.knot_xs[proc]
			fixed[  proc, :, elem] = True

			# 4: Strongly require that q_X sum to 1 for Ia_elem
			proc = FixedParams.processes == "Ia"
			elem = AbundData.elements == FixedParams.Ia_elem
			Lambdas[proc, :, elem] = FixedParams.Lambda_a
			q0s[    proc, :, elem] = 1.0 - (FixedParams.q_CC_Fe + FixedParams.dq_CC_Fe_dZ * FixedParams.knot_xs[proc])
			fixed[  proc, :, elem] = True

			self._init_q = init_q
			self._Lambdas = Lambdas
			self._q0s = q0s
			self._lnq0s = np.log(np.clip(q0s, 1.e-7, None))
			self._fixed = fixed

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
		def fixed(self):
			return self._fixed
	
	

