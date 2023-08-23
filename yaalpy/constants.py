import torch
DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

HEIGTH = 512
WIDTH = 1024

OBJECT_RADIUS = 10

QUADTREE_MAX_CAPACITY = 1

"""
Possible brain possible :

YaacBrainVector
YaacBrainMLP
YaacBrainCNN
"""

DEFAULT_BRAIN = 'YaacBrainMLP'

MAX_FOV = 101

DEFAULT_NB_WORLD_CHANNELS = 9

RGB_CHANNELS = [0, 1, 2]            #R, G, B
IR_UV_CHANNELS = [3, 4]             #IR, UV
PHEROMONE_CHANNELS = [5, 6, 7, 8]   #Methane, Minthol, Ethanol, Butanol

PHEROMONE_DIFFUSION_STRENGTH = 1
PHEROMONE_EVAPORATION_FACTOR = 0.97

DEFAULT_WORLD_IMPRINT_FOOD = torch.tensor([
    [#R
        [0, 0, 0.25, 0, 0],
        [0, 0.5, 0.78, 0.5, 0],
        [0.25, 0.78, 0.78, 0.78, 0.25],
        [0, 0.5, 0.78, 0.5, 0],
        [0, 0, 0.25, 0, 0],
    ],
    [#G
        [0, 0, 0.1, 0, 0],
        [0, 0.2, 0.3, 0.2, 0],
        [0.1, 0.3, 0.3, 0.3, 0.1],
        [0, 0.2, 0.3, 0.2, 0],
        [0, 0, 0.1, 0, 0],
    ],
    [#B
        [0, 0, 0, 0, 0],
        [0, 0, 0.235, 0, 0],
        [0, 0.235, 0.235, 0.235, 0],
        [0, 0, 0.235, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [#IR
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [#UV
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [#Methane
        [0, 0.1, 0.2, 0.1, 0],
        [0.1, 0.2, 2, 0.2, 0.1],
        [0.2, 2, 5, 2, 0.2],
        [0.1, 0.2, 2, 0.2, 0.1],
        [0, 0.1, 0.2, 0.1, 0]
    ],
    [#Minthol
        [0, 0.01, 0.02, 0.01, 0],
        [0.01, 0.02, 0.2, 0.02, 0.01],
        [0.02, 0.2, 0.5, 0.2, 0.02],
        [0.01, 0.02, 0.2, 0.02, 0.01],
        [0, 0.01, 0.02, 0.01, 0]
    ],
    [#Ethanol
        [0, 0.01, 0.02, 0.01, 0],
        [0.01, 0.02, 0.2, 0.02, 0.01],
        [0.02, 0.2, 0.5, 0.2, 0.02],
        [0.01, 0.02, 0.2, 0.02, 0.01],
        [0, 0.01, 0.02, 0.01, 0]
    ],
    [#Butanol
        [0, 0.1, 0.2, 0.1, 0],
        [0.1, 0.2, 1, 0.2, 0.1],
        [0.2, 1, 2, 1, 0.2],
        [0.1, 0.2, 1, 0.2, 0.1],
        [0, 0.1, 0.2, 0.1, 0]
    ],
])