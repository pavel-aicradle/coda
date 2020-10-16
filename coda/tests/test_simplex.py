# run with python3 -m pytest

import pytest
import time
import numpy
from ..coda import simplex


def test_simplexes_are_regular():
	for i in range(2,6):

		# call the function and see how long it takes
		a = time.time()
		s = simplex(i)
		print(s)
		b = time.time()
		print("simplex("+str(i)+") found in "+str((b-a)*1000)+" ms")

		# we want the norms of all points to be 1, except for the origin, which has norm zero
		norms = numpy.linalg.norm(s, axis=1)
		assert norms[0] == 0
		assert sum(norms[1:]) - (i-1) < 1e-8

		# double check that all pairs of points are distance 1 away from each other
		for p in range(len(s)):
			for q in range(len(s)):
				if p != q:
					assert abs(numpy.linalg.norm(s[p]-s[q]) - 1) < 1e-8
				else:
					assert numpy.linalg.norm(s[p]-s[q]) == 0
