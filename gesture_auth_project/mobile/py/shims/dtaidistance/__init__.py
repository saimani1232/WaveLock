"""NumPy stand-in for the parts of dtaidistance WaveLock uses (Pyodide has no
build of its C extension). Checked against the real package by
mobile/tests/test_mobile_core.py."""
from . import dtw_ndim  # noqa: F401
