
import numpy
from typing import Union, Tuple

class CODA:
	"""Performs outlier detection on a graph with respect to fellow community members. That is, clusters and finds
	outliers in the graph simultaneously. Outliers are not global, are not with respect to only local neighbors, and
	are not based communities found via clustering alone, which can be thrown off by the presence of anomalous nodes.
	"""

	def __init__(self, S: list, W: numpy.ndarray, K: int, lambda_: float, n_outliers: Union[int, float]=10):
		"""Constructor

		:param S: A list of data objects, one per node, including all relevant observable data for that node. Nodes
			are implicitly referred to by index, encoded as their order in this list.
		:param W: A weighted adjacency matrix encoding which nodes are connected and how strongly. Nodes should be ordered
			same as in S.
		:param K: The number of communities to form in the graph
		:param lambda_: How strongly to weight neighbors' assignments in consideration of a node's assignment. Smaller
			lambda_ means more reliance on a node's own attribute data and less on links.
		:param n_outliers: Set this parameter to an integer if you want to find a specific number of outliers; set it to
			a float in [0, 1] to assume the corresponding percentage of nodes are anomalous
		"""
		# Set all those params to be attributes of the object, without having to do each one individually
		for k, v in locals().items():
			if k != 'self':
				setattr(self, k, v)

		# based on what is in S, I have to decide how to model the observable random variables X
		self.P = ?
		self.Theta = ?

		Z_prev = # random
		Z_t = # be careful
		t = 1
		while Z_t #not close enough to Z_prev: # norm?
			#M_step
			#E_step
			t += 1
		return # indices of outliers


	def _icm(self):
		"""Iterated Conditional Modes
		"""
		Z_prev = # random
		U = numpy.zeros(len(S)) # keep track of M (= number of nodes) energy values
		while Z_t # not close enough to Z_prev:
			Z_prev = Z_t
			for i in range(len(S)):
				Z_t[i], U[i] = self._energy_argmin(Z_t, i)

			# Have to order the energies and choose top n or top % as outliers
		return Z_t

	def _energy_argmin(self, Z_t: numpy.ndarray, i: int) -> Tuple[int, float]:
		"""find the answer to equation 6 from the paper

		:param Z_t: The current assignment of nodes, which _icm is trying to refine
		:param 
		"""
		best_k = None
		best_U = float('inf')

		# used in calculation of neighbor contribution, unchanging for all k, so calculate once outside the loop
		W_i = numpy.copy(self.W[i])
		W_i[i] = 0 # if there happens to be a self-connection, zero its contribution to this

		for k in range(1, self.K+1): # for k in 1..K
			# Find contribution from the community neighbors. If node i is an outlier, it has no neighbors. Otherwise
			# its neighbors are the set {j : w_ij > 0, i != j, z_j != 0}
			neighbors_contribution = 0 if Z_t[i] == 0 else numpy.dot(W_i, k - Z_t == 0) # W_i dot delta(k-Z)

			# find self contribution
			self_contribution = self.P(self.S[i], self.Theta[k])

			U = -numpy.log(self_contribution) - self.lambda_*neighbors_contribution

			if U < best_U:
				best_U = U
				best_k = k
		
		return best_k, best_U






