import pandas
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.structure.graph import Graph
from gremlin_python.process.traversal import T
from numpy import ndarray

from coda import CODA

graph = Graph()
remoteConn = DriverRemoteConnection('wss://sensing-tf3.cluster-cfzqaib9mklx.us-west-2.neptune.amazonaws.com:8182/gremlin','g')

def getPersons():
    g = graph.traversal().withRemote(remoteConn)
    nodes = g.V().hasLabel('dummy_person').valueMap(True).toList()
    return {x[T.id]: {'personId': x['person_id'][0], 'name': x['name'][0], 'income': x['income'][0]} for x in nodes}

def getKnows(vertexId):
    g = graph.traversal().withRemote(remoteConn)
    return g.V(vertexId).out('knows').id().toList()

def getCommunity(vertexId):
    g = graph.traversal().withRemote(remoteConn)
    return g.V(vertexId).out('member_of').id().toList()[0]

def getMappedNodes():
    # get all individual person nodes
    mappedNodes = getPersons()
    for nodeId in mappedNodes:
        knowsIds = getKnows(nodeId)
        mappedNodes[nodeId]['knowsIds'] = knowsIds
        mappedNodes[nodeId]['communityId'] = getCommunity(nodeId)
    return mappedNodes

def getAdjacencyMatrix(nodes):
    nodeList = [x for x in nodes]
    matrix = []
    for x in range(len(nodes)):
        matrix.append([0 for x in range(len(nodes))])
    for nodeId in nodes:
        selfIndex = nodeList.index(nodeId)
        for knowsId in nodes[nodeId].get('knowsIds', []):
            knowsIndex = nodeList.index(knowsId)
            matrix[selfIndex][knowsIndex] = 1
    return matrix

def main():
    nodes = getMappedNodes()
    matrix = getAdjacencyMatrix(nodes)
    data_frame = pandas.DataFrame(nodes)
    adjacencyMatrix = ndarray(matrix)
    coda = CODA(data_frame, adjacencyMatrix, 3, 0.5)
    remoteConn.close()


if __name__ == "__main__":
    main()





