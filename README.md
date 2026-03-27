# Locally Adaptive Decay Surfaces (LADS)

Python package for event-based vision processing with LADS, paper preprint available [here](https://arxiv.org/abs/2602.23101).


![Demo](https://github.com/C3Imaging/LADS/blob/main/assets/face_demo.gif?raw=true "LADS Demo GIF")

## Installation
You can install the package using:
```bash
pip install event-lads
```

## Usage
The `LADS` class is the core of this package and is used to integrate event data into locally adaptive decay surfaces. See the code snippet below for sample usage. 
A full implementation for all decay functions with a [real event clip](https://github.com/C3Imaging/LADS/blob/main/examples/face.npy) is provided in [create_event_video.py](https://github.com/C3Imaging/LADS/blob/main/examples/create_event_video.py).

```python
import torch
import numpy as np
from event_lads import *
import matplotlib.pyplot as plt

# Initialize LADS
lads = LADS(
    H=64, W=64, # Spatial dims of input events and output surface
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    decay_func="er", # Options:["global-li", "er", "fft", "log"]
    reference_event_rate=0.1,
    decay_param=0.2,
    patch_size=8
)

# Generate dummy event data 
events = np.array([
    [0.1,  32, 32,  1], #(timestamp, x, y, polarity)
    [0.11, 10, 20,  1],
    [0.2,  40, 50, -1],
    [0.3,  32, 32,  1],
    [0.35, 32, 32,  1],
    [0.36, 10, 20,  1],
    [0.4,  40, 50, -1],
    [0.5,  32, 32,  1]
])


# Generate the surface by integrating the first block of events:
surface, patch_scores, patch_decay_factors = lads.integrateEvents(events[:4]) 
# patch_scores & patch_decay_factors are no longer needed but returned for analysis/visualisation.


frame = LADS_to_output_frame(surface, clip_val=3) # Convert the surface (tensor) to frame (ndarray).
plt.imshow(frame, cmap='gray')
plt.show()

# Update the surface by integrating more events:
surface, patch_scores, patch_decay_factors = lads.integrateEvents(events[4:]) 

frame = LADS_to_output_frame(surface, clip_val=3)
plt.imshow(frame, cmap='gray')
plt.show()
```

