"""Duple-balanced ensemble learning on class-imbalanced dataset.
"""

from .duple_balance import DupleBalanceClassifier
from . import sampler
from . import utils
from ._version import __version__

__all__ = [
    "DupleBalanceClassifier",
    "sampler",
    "utils",
    "__version__",
]