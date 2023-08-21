import torch
DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

HEIGTH = 500
WIDTH = 1280

OBJECT_RADIUS = 10

QUADTREE_MAX_CAPACITY = 4

"""
Possible brain possible :

YaacBrainVector
YaacBrainMLP
YaacBrainCNN
"""

DEFAULT_BRAIN = 'YaacBrainMLP'

MAX_FOV = 101

DEFAULT_NB_WORLD_CHANNELS = 9

RGB_CHANNELS = [0, 1, 2]
IR_UV_CHANNELS = [3, 4]
PHEROMONE_CHANNELS = [5, 6, 7, 8]

PHEROMONE_DIFFUSION_STRENGTH = 1
PHEROMONE_EVAPORATION_FACTOR = 0.97