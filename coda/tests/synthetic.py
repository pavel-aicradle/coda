import numpy
import pandas
from scipy.stats import powerlaw
from matplotlib import pyplot

def synthetic(n, m, n_outliers, outlier_type, symmetric=True, show_plots=False):
	"""Generate a large synthetic graph.

	:param n: how many nodes per community
	:param m: how many communities
	:param n_outliers: how many outliers to find
	:param outlier_type: {'far_away', 'connections', 'swap_communities'}
	:param symmetric: Whether to make the graph symmetric
	:param show_plots: Whether to show plots to check some distributions of the data
	"""
	# create the dataframe of features
	data = []
	for i in range(m):
		cov = numpy.random.normal(size=(m,m))*numpy.random.normal(size=m)
		cov = cov.dot(cov.T)/m # makes positive semidefinite
		mean = numpy.random.normal(size=m)*numpy.random.uniform(low=-10, high=10, size=m)
		data.append(numpy.random.multivariate_normal(mean, cov, n)) # draw n samples
	S = pandas.DataFrame(data=numpy.concatenate(data), columns=range(m))

	# create connections between people
	W_dict = {i:[] for i in range(len(S))}
	for i in range(len(S)):
		# create probability distribution for selecting friends
		p = numpy.ones(len(S))
		community = slice(i//n*n,(i//n+1)*n)
		p[community] += abs(numpy.random.normal(size=n)*5) # more likely to be friends with own community
		p[W_dict[i]] = 0 # for friends I already have, I don't need to reselect them.
		p[i] = 0 # no self-connections
		p = p/sum(p) # normalize

		# everyone has at least one friend, but subtract off the friends one already has
		n_new_friends = max(int(powerlaw.rvs(0.3)*len(S)) - len(W_dict[i]), 1 if not W_dict[i] else 0)
		new_friends = numpy.random.choice(range(len(S)), size=n_new_friends, replace=False, p=p)

		# assign friends, keeping track of reciprocal connections if symmetric
		W_dict[i] += list(new_friends)
		if symmetric:
			for j in new_friends:
				W_dict[j].append(i)

	# Go from person -> friends map to adjacency matrix
	W = numpy.zeros((len(W_dict), len(W_dict)))
	for i in W_dict:
		for j in W_dict[i]:
			W[i-1][j-1] = 1 if symmetric else abs(numpy.random.normal())

	#Plots to check the above is right
	if show_plots:
		# Should show power law distribution
		yo = [sum(x) for x in W]
		pyplot.hist(yo, bins=30)
		pyplot.title('power law distribution')
		pyplot.ylabel('number of nodes with number of friends in each bucket')
		pyplot.xlabel('number of friends a node has')
		pyplot.show()

		# Should show higher than random concentration of friends in own community
		yo = []
		for i in range(len(S)):
			friends = W[i]
			community = slice(i//n*n,(i//n+1)*n)
			yo.append(sum(friends[community])/sum(friends))
		pyplot.hist(yo, bins=30)
		pyplot.title("portion of friends that come from nodes' own communities")
		pyplot.xlabel('fraction of friends from same community')
		pyplot.ylabel('number of nodes with each fraction')
		pyplot.show()

	# corrupt some of the features and connections to make nodes anomalous
	outlier_ndxs = []
	if outlier_type == 'far_away':
		for i in numpy.random.choice(range(len(S)), size=n_outliers, replace=False):
			
			S.iloc[i] = 10 + numpy.random.normal(size=m)*5 # clearly not near any of the distributions
			outlier_ndxs.append(i)

	elif outlier_type == 'connections':
		for j in range(n_outliers):
			i = numpy.random.choice(range(len(S)))
			community = slice(i//n*n,(i//n+1)*n)

			W[i,:] = 2 # connect to everyone
			W[:,i] = 2 # connect everyone back
			W[i,community] = 0.5 # weaken connections to own group
			W[community,i] = 0.5 # weaken connections from own group
			outlier_ndxs.append(i)

	elif outlier_type == 'swap_communities':
		if n_outliers % 2 != 0: raise ValueError('n_outliers needs to be even if outlier_type is swap_communities')
		temp = numpy.zeros(m)
		for j in range(n_outliers//2):
			i1 = numpy.random.choice(range(len(S)))
			community = slice(i//n*n,(i//n+1)*n)
			p = numpy.ones(len(S))
			p[community] = 0
			p = p/sum(p)
			i2 = numpy.random.choice(range(len(S)), p=p) # guarantee i2 is from different community
			temp[:] = S.iloc[i1]
			S.iloc[i1] = S.iloc[i2]
			S.iloc[i2] = temp
			outlier_ndxs += [i1, i2]

	else:
		raise ValueError('not a valid choice of outlier_type')

	return S, W, outlier_ndxs
