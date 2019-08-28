"""
Unit and regression test for the stubs package.
"""

# Import package, test suite, and other packages as needed
import stubs
import pytest
import sys

def test_stubs_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "stubs" in sys.modules
