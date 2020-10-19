# run with python3 -m pytest -s

import pytest
import numpy
import pandas
from ..coda import CODA
from ..neptune import getMappedNodes, getAdjacencyMatrix


def test_coda_on_paper_example():

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

	model = CODA(S, W, K=2, lambda_=0.5, n_outliers=1, generating_distribution='independent',
		return_all=True)

	node, energy = model.run()

	print("node: ", node)
	print("energy: ", energy)
	print("sum of energy: ", sum(energy))

	print("made it here")

def test_coda_on_dummy_data():
	nodes = getMappedNodes()
	matrix = getAdjacencyMatrix(nodes)
	S = pandas.DataFrame({'income': [nodes[x]['income'] for x in nodes]})
	W = numpy.array(matrix)
	model = CODA(S, W, K=2, lambda_=0.5, n_outliers=1, generating_distribution='independent',
				 return_all=True)

	node, energy = model.run()

	print("node: ", node)
	print("energy: ", energy)
	print("sum of energy: ", sum(energy))

	print("made it here")