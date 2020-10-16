import numpy

def simplex(n) -> numpy.ndarray:
	"""Return the n vertices of a regular (n-1)-simplex, which lives in (n-1) dimension. I'm choosing one of the
	vertices to always be the origin and for distances between vertices to be 1. There's a simple recurrence
	relation between the k-simplex and the (k-1)-simplex, so it's possible to build up to n-1 iteratively.

	:param n: the number of vertices the user would like
	:returns: a 2D numpy array, where each row is the coordinates of a vertex
	"""
	if n < 2: raise ValueError('This function is intended to work only for the 1-simplex and above.')

	vertices = numpy.zeros((n, n-1)) # there are n vertices, each living in dimension n-1

	# first coordinate is always the origin, a bunch of zeros, so nothing to change
	for i in range(1, n): # build up to the (n-1) simplex using a growing portion of the table
		for j in range(i-1): # iterate across the existing coordinates, where the new point is the centroid
			vertices[i,j] = numpy.mean(vertices[:i,j]) # mean of all previous points in this dimension
		vertices[i,i-1] = numpy.sqrt(1 - numpy.linalg.norm(vertices[i,:i-1])**2)

	return vertices
