""" Detectors

A collection of detectors that act as a realistic filter to the raw output
power of the fields.

"""

from .lowpassdetector import lfilter
from .lowpassdetector import LowpassDetector
from .photodetector import Photodetector
