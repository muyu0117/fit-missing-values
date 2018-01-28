import rpy2.tests
import unittest

tr = unittest.TextTestRunner(verbosity = 1)
suite = rpy2.tests.suite()
tr.run(suite)
import pdb; pdb.set_trace()  # breakpoint 2af9f81f //
print "done"
