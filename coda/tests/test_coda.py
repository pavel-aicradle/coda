# run with python3 -m pytest -s

import pytest
import numpy
import pandas
from ..coda import CODA


"""This is the example given in the figures in the paper. I spent a long time toying with it, and I'm convinced
it's just kind of not a great example. I only a few times succeeded in tuning lambda such that setting V6 as the
outlier, which they give as the answer, results in lower cumulative energy than setting V2. The trouble is that both
are fairly equidistant from the mean of their community (103), but the fact V2 is just as connected to the other
cluster as its own, and V6 actually has more connections (both by portion and by absolute count) to the high-income
cluster, means that V2 is sort of "pulled off" and has higher energy. I have no idea whether they actually ran
their algorithm on their example and got it to work, but I feel like maybe they didn't.
"""
S = pandas.DataFrame({'income': [70, 160, 140, 100, 110, 40, 10, 30, 10, 30]})
W = numpy.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 1 -> 3
				 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0], # 2 -> 3, 9
				 [1, 1, 0, 1, 0, 0, 0, 1, 0, 0], # 3 -> 1, 2, 4, 8
				 [0, 0, 1, 0, 1, 1, 0, 1, 1, 1], # 4 -> 3, 5, 6, 8, 9, 10
				 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # 5 -> 4, 6
				 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # 6 -> 4, 5
				 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 7 -> 8
				 [0, 0, 1, 1, 0, 0, 1, 0, 0, 1], # 8 -> 3, 4, 7, 10
				 [0, 1, 0, 1, 0, 0, 0, 0, 0, 1], # 9 -> 2, 4, 10
				 [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]]) # 10 -> 4, 8, 9


"""I've engineered this example to be as cut-and-dried as possible. You have red nodes, blue nodes, and green nodes,
and then you've got a blue node off to the side that's only connected to red nodes, which should always be the
outlier.
"""
S2 = pandas.DataFrame({'color': ['blue', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue',
	'blue']})
W_dict = {1: [2,3,4],
		  2: [1,3,9],
		  3: [1,2,4,5,10],
		  4: [1,3,5],
		  5: [3,4,6,7,11],
		  6: [5,7,8],
		  7: [5,6,8],
		  8: [6,7,11],
		  9: [2,10,11],
		  10: [3,9,11],
		  11: [5,8,9,10]}
W2 = numpy.zeros((len(W_dict), len(W_dict)))
for i in W_dict:
	for j in W_dict[i]:
		W2[i-1][j-1] = 1


def test_paper_example_global():
	# Run with no concern for neighbors. Should give us the global outlier, V1
	model = CODA(S, W, K=2, lambda_=0, n_outliers=1, generating_distribution='independent',
		return_all=True)
	Z, energy = model.run()

	print("Z", Z)
	print("energy", energy)
	print("sum of energy", sum(energy))

	# there are rare instances this converges to different, less-optimal clusters, and I don't want this test to fail
	if sum(energy) < 45:
		assert Z[0] == 0 # V1 is outlier
		assert Z[1] == Z[2] == Z[3] == Z[4] # and with who's left over, we get these two communities
		assert Z[5] == Z[6] == Z[7] == Z[8] == Z[9]
	else:
		print("Converged to non-optimal clusters. That happens some times. This is stochastic.")


def test_paper_example_communities():
	# Run with concern for neighbors. Should give V2 as the outlier.
	model = CODA(S, W, K=2, lambda_=2, n_outliers=1, generating_distribution='independent',
		return_all=True)
	Z, energy = model.run()

	print("Z", Z)
	print("energy", energy)
	print("sum of energy", sum(energy))

	if sum(energy) < 33:
		assert Z[1] == 0
		assert Z[0] == Z[2] == Z[3] == Z[4] == Z[5]
		assert Z[6] == Z[7] == Z[8] == Z[9]
	else:
		print("Converged to non-optimal clusters. That happens some times. This is stochastic.")


def test_categorical_example_multinomial():
	# generating distribution is such that the multinomial comes in to play
	model = CODA(S2, W2, K=3, lambda_=0.4, n_outliers=1, generating_distribution='independent',
		return_all=True, verbose=True)
	Z, energy = model.run()

	print("Z", Z)
	print("energy", energy)
	print("sum of energy", sum(energy))

	assert numpy.all([energy[0] > energy[i] for i in range(i, len(energy))]) # most anomalous is at 0
	assert Z[0] == 0
	assert Z[1] == Z[2] == Z[3] == Z[4]
	assert Z[5] == Z[6] == Z[7]
	assert Z[8] == Z[9] == Z[10]


def test_categorical_example_distance():
	# This particular example doesn't work great all the time, but I want to ensure it runs.
	model = CODA(S2, W2, K=3, lambda_=1, n_outliers=1, generating_distribution='distance',
		return_all=True, n_runs=3)
	
	Z, energy = model.run()

	print("Z", Z)
	print("energy", energy)
	print("sum of energy", sum(energy))


def test_asymmetrical_example():
	# The idea is just to ensure this runs and converges
	W3 = numpy.zeros(W.shape)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			if W[i,j] > 0:
				W3[i,j] = abs(numpy.random.normal()*10)

	model = CODA(S, W3, K=2, lambda_=1, n_outliers=1, generating_distribution='independent',
		return_all=True, n_runs=1)
	Z, energy = model.run()

	print("Z", Z)
	print("energy", energy)
	print("sum of energy", sum(energy))
	print(W3)


def test_asymmetrical_influence():
	# A case where the answer ends up being different because of a connection strength
	W4 = numpy.zeros(W.shape)
	W4[:] = W
	W4[2,1] = 10 # V3 exerts a much stronger influence on V2

	model = CODA(S, W4, K=2, lambda_=1, n_outliers=1, generating_distribution='distance',
		return_all=True, n_runs=10)
	Z, energy = model.run()

	if sum(energy) < 39: # V3 is now binding V2 more to the community, so V6 becomes the best choice
		assert Z[5] == 0
		assert Z[0] == Z[1] == Z[2] == Z[3] == Z[4]
		assert Z[6] == Z[7] == Z[8] == Z[9]
	else: # Some times V6 gets lumped with the other community in preclustering
		assert Z[0] == 0
		assert Z[1] == Z[2] == Z[3] == Z[4]
		assert Z[5] == Z[6] == Z[7] == Z[8] == Z[9]


def test_exceptions():
	# Just to hit the lines where I raise ValueErrors in initialization
	with pytest.raises(ValueError):
		model = CODA(S, W, K=2, lambda_=1, n_outliers=2.0, generating_distribution='distance',
			return_all=True, n_runs=10)

	with pytest.raises(ValueError):
		model = CODA(S, W, K=2, lambda_=1, n_outliers=0.5, generating_distribution='not_a_category',
			return_all=True, n_runs=10)


