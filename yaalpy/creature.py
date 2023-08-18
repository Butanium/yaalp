import numpy as np
import torch
from torch import nn

from constants import *
from world import World

class MovingObject:
    def __init__(self, x=0, y=0, speed=0, direction=0):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
    
    def move(self):
        self.x += self.speed * np.cos(self.direction)
        self.y += self.speed * np.sin(self.direction)
    
    def set_speed(self, speed):
        self.speed = speed
    
    def set_direction(self, direction):
        self.direction = direction
    
    def set_position(self, x, y):
        self.x = x
        self.y = y


class YaacStats:
    def __init__(self,
            health=100,
            max_health=100,
            energy=100,
            max_energy=100,
            hunger=0,
            max_hunger=100,
            max_speed=100,
            age=0
        ) -> None:
        self.health = health
        self.max_health = max_health
        self.energy = energy
        self.max_energy = max_energy
        self.hunger = hunger
        self.max_hunger = max_hunger
        self.max_speed = max_speed
        self.age = age


class YaacBrainVector(nn.Module):
    """
    A simple vector of NbWorldChannel used to make a product with each pixel in the fov, and update the speed based on the result.
    This is very light in computations but has no non-linearity and may limit the Yaac's ability to learn.
    """
    
    #The Module class is not used to train the brain, but to make it easier to use the GPU.

    def __init__(self, world: World):
        super().__init__()
        self.decision_vectors = nn.Parameter(torch.randn(world.nb_possible_actions, world.nb_world_channels))
    
    def forward(self, input):
        # input has shape (fov, fov, nb_world_channels)
        # decision_vector has shape (nb_possible_actios, nb_world_channels)
        # output has shape (fov, fov)
        # The output map will be used for decision making.
        return torch.einsum('ijk,ak->ija', input, self.decision_vectors)


class YaacBrainCNN(nn.Module):
    """
    This brain is a CNN that takes as input the square surrounding the Yaac.
    It is a basic CNN with 3 convolutional layers and 2 fully connected layers.
    For now, the output is an acceleration on the angle and norm of the speed vector.
    Future versions may include more complex outputs.

    This brain is very elegant as it gives sense to... senses, like vision, smell, pheromones, etc.
    However, it will be very difficult to include hunger, pain, etc. in the input. The other brains may be better for that.
    """
    def __init__(self, world: World):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv2d(world.nb_world_channels, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(3, 3),
        # TODO
        )
        pass

class YaacBrainMLP(nn.Module):
    """
    This brain architecture is an attempt to reinclude internal states in the brain.
    This brain will be applied like the vector brain, on each pixel,
        and the actions are decided based on the result (maybe the mean, max, or other aggregation).
    Each pass will need the NbWorldChannel inputs from the current pixel, as well as the internal states like hunger, pain, etc.

    The idea is that if you are very hungry and your life is very low, when you see a predator you don't want to go fight it,
        and if you see food, you absolutely want to go eat it. This way, decisions will not be taken based only on the vision
        of the external world, but will also take into account the Yaac's internal states.
        Furthermore, as we use an MLP, we will have more advanced decisions than with the simple vector brain.

    It is also not so bad in terms of computations as we can flatten the fov and make it a big batch to pass through the MLP.
    """
    def __init__(self, world: World, nb_internal_states):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(world.nb_world_channels + nb_internal_states, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

class YaacGenome:
    # The genome may include other parameters in the future, like brain architecture.
    # For now, only the field of view is variable and the brain has a fixed architecture.
    # This is easier to implement and to cross over.
    def __init__(self,
            world: World,
            # Field of View is in pixel. It measures the side of the square viewed by the Yaac.
            fov=3,
            max_health=100,
            max_energy=100,
            max_speed=100,
    ) -> None:
        self.fov = fov
        self.max_health = max_health
        self.max_energy = max_energy
        self.max_speed = max_speed
        self.brain = YaacBrain(world)

# Yet Another Artificial Creature.
# The Yaac is the base class for all creatures in the game.
class Yaac(MovingObject):
    def __init__(self,
            world: World,
            x=0,
            y=0,
            speed=0,
            direction=0,
            sprite="assets/default_yaac.png"
        ) -> None:
        super().__init__(x, y, speed, direction)

        self.stats = YaacStats(world)
        self.genome = YaacGenome(world)

        # This is the number of attributes in YaacStats.
        self.nb_internal_states = 8
        self.nb_decisions

        self.sprite = sprite