import logging

from .core import Solver

__version__ = "0.2.0"

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())