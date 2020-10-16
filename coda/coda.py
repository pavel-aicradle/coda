
import numpy
import pandas
from typing import Union, Tuple
from .simplex import simplex

class CODA:
	"""Performs outlier detection on a graph with respect to fellow community members. That is, clusters and finds
	outliers in the graph simultaneously. Outliers are not global, are not with respect to only local neighbors, and
	are not based communities found via clustering alone, which can be thrown off by the presence of anomalous nodes.
	"""

	def __init__(self, S: pandas.DataFrame, W: numpy.ndarray, K: int, lambda_: float, n_outliers: Union[int, float]=10,
		generating_distribution: str='independent'):
		"""Constructor

		:param S: A table, with one row per node, including all relevant observable data for that node. Nodes are
			implicitly referred to by index, encoded as their order in this table.
		:param W: A weighted adjacency matrix encoding which nodes are connected and how strongly. Nodes should be
			ordered same as in S.
		:param K: The number of communities to form in the graph
		:param lambda_: How strongly to weight neighbors' assignments in consideration of a node's assignment. Smaller
			lambda_ means more reliance on a node's own attribute data and less on links.
		:param n_outliers: Set this parameter to an integer if you want to find a specific number of outliers; set it to
			a float in [0, 1] to assume the corresponding percentage of nodes are anomalous
		:param generating_distribution: {'independent','distance'} a parameter to control how CODA models the process
			that gives rise to the observable attribute data. 'independent' assumes that all attributes arise
			independently, where continuous variables are sampled from Gaussians and categorical variables are sampled
			from multinomials, and the joint distribution defining a community is the product of these processes.
			'distance' instead defines communities using a measure of centrality and a measure of distance to that
			center (tricky when you include categorical variables). The joint distribution then becomes e^-distance.
			For distance I'm using Mahalanobis squared, which measures "the dissimilarity of two random vectors x and y
			of the same distribution with covariance matrix Sigma".
		"""
		# Sanity checking parameters
		if not (isinstance(n_outliers, float) and n_outliers > 0 and n_outliers < 1) \
			and not isinstance(n_outliers, int):
			raise ValueError('n_outliers must be an int or a float in [0,1]')
		if not generating_distribution in ['independent', 'distance']:
			raise ValueError("generating_distribution should be in {'independent', 'distance'}")

		# Set all those params to be attributes of the object, without having to do each one individually
		for k, v in locals().items():
			if k != 'self':
				setattr(self, k, v)

		# Based on what is in S and the generating distribution specified, I have to decide how to model the observable
		# random variables X with distributions P and parameters Theta.

		# First step is to infer which attributes in S are numerical and which are categorical. For categoricals, it
		# becomes important to know how many possible settings each one has, and for the 'distance' case it's important
		# each one have a specific index
		self.is_categorical = (S.dtypes == 'category') | (S.dtypes == 'O') # pandas Series
		self.categorical_ndx = {}
		for col, v in is_categorical.iteritems():
			if v: categorical_ndx[col] = dict({x:i for i,x in enumerate(S[col].unique())}) # map unique values -> indices

		# Next, define the proper distribution. That means both the function and the parameters that lock its shape.
		if generating_distribution=='independent': # ln(P(A,B,C)) = ln(P(A)*P(B)*P(C)) = ln(P(A)) + ln(P(B)) + ln(P(C))
			self.logP = self.__gaussians_and_multinomials
			# Theta[k] = parameters for the kth community. kth community params = a dictionary of column name ->
			# parameters for that attribute. Parameters for a categorical column is a further dictionary of choice -> %.
			# Parameters for a numerical column are (mu, sigma^2)
			self.Theta = {col: {} if is_categorical[col] else None for col in S} # will get filled at first Maximization step
		else: # ln(P(A,B,C)) = ln(e^distance((A,B,C) - mu)) = distance((A,B,C) - mu)
			self.logP = self.__radial_basis_function

			self.Theta = None # not sure yet
			

	def run(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Run the optimization procedure and return an answer
		
		:returns: The indices and energies (relative anomalousness measure) of outliers
		"""

		Z_prev = numpy.zeros(len(self.S)) # the point is just to be far from Z_t for first while loop condition check
		Z_t = numpy.random.choice(numpy.arange(1, self.K+1), size=len(self.S)) # random assignment
		# be careful. One choice is just to run the algorithm multiple times with different initialization here.

		while numpy.mean(Z_t != Z_prev) > 0.1: # Z_t not close enough to Z_prev. Here I'm saying no more than 10% can
			#have different class. Is there a better cutoff for this iteration?
			Z_prev = Z_t

			# M_step: choose model parameters for generating distribution(s), assuming assignment
			self._Q_argmax(Z_t)
			
			# E_step: choose assignment, assuming model parameters
			Z_t, U = self._icm(Z_t)

		outlier_ndxs = numpy.where(Z_t == 0)
		return outlier_ndxs, U[outlier_ndxs] # indices of outliers + energies


	def _icm(self, Z_t: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""'Iterated Conditional Modes is a deterministic algorithm for obtaining a configuration of a local maximum of
		the joint probability of a Markov random field. It does this by iteratively maximizing the probability of each
		variable conditioned on the rest.' -Wikipedia. Basically you're looking for the best assignment of Z assuming
		your generating distributions of X=S are right.

		:param Z_t: the current assignment of node clusters
		:returns: newly optimized assignment of node clusters, corresponding assignment energies
		"""

		def _energy_argmin(self, Z_t: numpy.ndarray, i: int) -> Tuple[int, float]:
			"""helper function to find the answer to equation 6 from the paper https://cse.buffalo.edu/~jing/doc/kdd10_coda.pdf,
			which optimizes a single mode

			:param Z_t: The current assignment of nodes, which _icm is trying to refine
			:param i: the index of the current mode, to be refined at this iteration
			:returns: the index k of the best cluster to explain the ith node
			"""
			best_k = None
			best_U = float('inf')

			# used in calculation of neighbor contribution, unchanging for all k, so calculate once outside the loop
			W_i = numpy.copy(self.W[i])
			W_i[i] = 0 # if there happens to be a self-connection, zero its contribution to this

			for k in range(1, self.K+1): # for k in 1..K, "for all clusters"
				# Find contribution from the community neighbors. If node i is an outlier, it has no neighbors. Otherwise
				# its neighbors are the set {j : w_ij > 0, i != j, z_j != 0}
				neighbors_contribution = 0 if Z_t[i] == 0 else numpy.dot(W_i, k - Z_t == 0) # W_i dot delta(k-Z)

				# Find self contribution, which is the log of the probability the data arose from the generating
				# distribution of the kth community.
				self_contribution = self.logP(self.S.iloc[i], self.Theta[k])

				U = -self_contribution - self.lambda_*neighbors_contribution

				if U < best_U:
					best_U = U
					best_k = k
			
			return best_k, best_U


		Z_prev = numpy.zeros(Z_t.shape) # the point is just to be far from Z_t for first while loop condition check
		U = numpy.zeros(len(S)) # keep track of all M (= the number of nodes) energy values

		while numpy.mean(Z_t != Z_prev) > 0.1: # Z_t not close enough to Z_prev. Here I'm saying no more than 10% can
			#have different class. Is there a better cutoff for this iteration?

			Z_prev = Z_t
			for i in range(len(S)):
				Z_t[i], U[i] = _energy_argmin(Z_t, i)

			# Have to order the energies and choose top n or top % as outliers
			top_n =	numpy.argsort(U)[-self.n_outliers:] if isinstance(self.n_outliers, int) \
				else numpy.argsort(U)[-self.n_outliers*len(U):]
			Z_t[top_n] = 0 # set the class of the top_n as 0, which means outlier

		return Z_t, U


	def _Q_argmax(self, Z_t: numpy.ndarray):
		"""The Maximization step of Expectation Maximization involves assuming the assignment Z is right and plugging
		it in to a variant, called Q, of the likelihood funtion (which relates parameters, observable, and hidden
		variables) that takes advantage of the dataset to sum out the hidden variables. Basically, your Z assignment
		(or probability distribution describing a probabilistic assignment in some cases), can be used as the pdf for
		Z in Q, and then you can plug in your generator distribution, take a derivative with respect to generator model
		parameters, set equal to zero, and solve. This doesn't simplify in all cases, but for the generator models
		I've chosen to allow, optimal parameter updates have closed-form solutions. This function simply makes those
		updates. http://homes.sice.indiana.edu/yye/i529/lectures/EM.pdf
		http://poseidon.csd.auth.gr/papers/PUBLISHED/JOURNAL/pdf/Ververidis08c.pdf

		:param Z_t: the current assignment of nodes to clusters
		"""
		if self.generating_distribution=='independent': # ln(P(A,B,C)) = ln(P(A)*P(B)*P(C)) = ln(P(A)) + ln(P(B)) + ln(P(C))

			for k in range(1, self.K+1): # for k in 1..K, "for each cluster"
				# Inside here we're really only going to be operating over nodes with class k
				community = S.iloc[Z_t == k] # dataframe only of nodes belonging to the kth cluster. Makes a copy.

				for col,meow in self.is_categorical.iteritems():
					if meow: # If the attribute in this column is a cat (categorical), then we're dealing with a multinomial.
						for choice in categorical_ndx[col]:
							self.Theta[k][col][choice] = sum(community[col] == choice)/len(community)
					else: # then we're modeling this attribute with a Gaussian
						self.Theta[k][col] = (community[col].mean(), community[col].var())

		else: # 'distance': # ln(P(A,B,C)) = ln(e^distance((A,B,C) - mu)) = distance((A,B,C) - mu)
			for k in range(1, self.K+1):
				community = S.iloc[Z_t == k] # This is where we're assuming Z_t is right.

				# TODO: find mean and covariance
				self.Theta[k] = (mu, Sigma)


	### generator distributions ###

	def __gaussians_and_multinomials(self, s_i: pandas.core.series.Series, theta_k: dict):
		"""This is for the independent case, where ln(P(A,B,C)) = ln(P(A)*P(B)*P(C)) = ln(P(A)) + ln(P(B)) + ln(P(C))
		Here in the numerical case: ln P(A=a | z=k, Theta) = ln univariate_gaussian(A, theta_k) = a one-line equation
		And in the categorical case: ln P(B=b | z=k, Theta) = ln P(B=b | theta_k) = ln theta_k[b] =
			ln |B=b|/|B=anything| = log of the parameter itself

		:param s_i: data corresponding to a particular node
		:theta_k: distribution parameters for the kth cluster
		:returns: the log probability that data s_i arose from a version of this distribution parameterized by theta_k
		"""
		log_p_sum = 0
		for col,meow in self.is_categorical.iteritems():
			if meow: # If the attribute in this column is a cat (categorical), then we're dealing with a multinomial.
				# The parameters of a multinomial are [beta_1, beta_2, ...], where beta_i = P(categorical = choice i)
				log_p_sum += numpy.log(theta_k[col][s_i[col]]) # theta_k[col] = {choice 1: beta_1, choice 2: beta_2 ...}
			else: # then we're modeling this attribute with a Gaussian
				# The paramerters of a Gaussian are (mu, sigma^2)
				mu, sigma2 = theta_k[col]
				log_p_sum += -numpy.log(numpy.sqrt(2*numpy.pi*sigma2)) - (s_i[col] - mu)**2/(2*sigma2)

		return log_p_sum


	def __radial_basis_function(self, s_i: pandas.core.series.Series, theta_k: dict):
		raise NotImplementedError('Coming Soon!') # TODO
		# I think I'd like to use the simplex method, which means I need some function to convert from categorical
		# to vertex coordinates


