"""
Testcases for the pulp.utils package
"""

import os.path
import tempfile
import shutil
import numpy as np
import unittest

import pulp.utils.tracing as tracing

#=============================================================================
# traceing tests

class TestTracing(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.fname = os.path.join(self.dirname, "trace-%04d.txt")
        tracing.set_tracefile(self.fname)

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_tracepoint(self):
        tracing.tracepoint("Test::begin")
        tracing.tracepoint("Test::end")

    def test_traced_wrapper(self):
        """Test that docs and name of traced functions are mantained."""
        @tracing.traced
        def funny_name(x, y, z):
            """Funny doc."""
            pass

        self.assertEqual(funny_name.__doc__, """Funny doc.""")
        self.assertEqual(funny_name.__name__, 'funny_name')
        
