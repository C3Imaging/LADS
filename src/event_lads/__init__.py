"""
Locally Adaptive Decay Surfaces Package

"""

# Import main classes
from .LADS import LADS

# Import event utilities
from .events_utils import (
    voxel,
    crop_events,
    pad_events,
    FixedSizeEventReader,
    FixedDurationEventReader,
    LADS_to_output_frame,
)

# Import recursive patch processing utilities
from .recursive_patches import (
    subdivide_grid_fast,
    subdivide_grid_recur,
)

# Define what gets exported when using "from lads import *"
__all__ = [
    # Main classes
    'LADS',
    'FixedSizeEventReader',
    'FixedDurationEventReader',
    
    # Event utilities
    'voxel',
    'crop_events',
    'pad_events',
    'LADS_to_output_frame',
    
    # Recursive patch utilities
    'subdivide_grid_fast',
    'subdivide_grid_recur',
]

# Package metadata
__version__ = '0.1'
__author__ = 'Paul Kielty'
__description__ = 'Locally Adaptive Decay Surfaces for event data representation'