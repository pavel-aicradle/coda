import numpy
import pandas

from coda import coda
from coda.neptune import getMappedNodes, getAdjacencyMatrix


def main():
    nodes = getMappedNodes()
    matrix = getAdjacencyMatrix(nodes)
    data_frame = pandas.DataFrame({'income': [nodes[x]['income'] for x in nodes]})
    adjacencyMatrix = numpy.array(matrix)
    model = coda.CODA(data_frame, adjacencyMatrix, K=2, lambda_=0.5, n_outliers=1, generating_distribution='independent', return_all=True)
    node, energy = model.run()

    print("node: ", node)
    print("energy: ", energy)
    print("sum of energy: ", sum(energy))


if __name__ == "__main__":
    main()