import numpy as np
import torch
from torch import nn

from constants import DEFAULT_BRAIN, DEVICE, HEIGTH, WIDTH, OBJECT_RADIUS, PHEROMONE_CHANNELS, RGB_CHANNELS, IR_UV_CHANNELS
from world import World
from graphics import Circle
import pygame as pg

class MovingObject:
    def __init__(self, x=WIDTH//2, y=HEIGTH//2, speed=0, direction=0):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
    
    def move(self):
        self.x += self.speed * np.cos(self.direction)
        self.y += self.speed * np.sin(self.direction)

        self.x = max(min(self.x, WIDTH - OBJECT_RADIUS), OBJECT_RADIUS)
        self.y = max(min(self.y, HEIGTH - OBJECT_RADIUS), OBJECT_RADIUS)
    
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
        ):
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

    def __init__(self, world: World, nb_internal_states):
        """
        nb_internal_states is not used here, but I need it to have the same interface as the other brains.
        See YaacGenome for more details.
        """
        super().__init__()
        self.decision_vectors = nn.Parameter(torch.randn(world.nb_possible_actions, world.nb_world_channels))
    
    def forward(self, input, internal_states):
        # input has shape (nb_world_channels, fov, fov)
        # decision_vector has shape (nb_possible_actios, nb_world_channels)
        # output has shape (fov, fov)
        # The output map will be used for decision making.
        return torch.einsum('chw,Cc->Chw', input, self.decision_vectors)


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
    def __init__(self, world: World, nb_internal_states, nb_world_channels=None):
        super().__init__()

        if nb_world_channels is None:
            nb_world_channels = world.nb_world_channels

        self.MLP = nn.Sequential(
            nn.Linear(nb_world_channels + nb_internal_states, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, world.nb_possible_actions)
        )
    
    def forward(self, pixel_input, internal_states):
        # pixel_input has shape (nb_world_channels, fov, fov)
        # internal_states has shape (nb_internal_states)
        # output has shape (nb_possible_actions)

        # We want to make a batch of size (fov*fov, nb_world_channels)
        fov = pixel_input.shape[1]
        x = pixel_input.reshape(-1, fov*fov)
        x = x.permute(1, 0)

        # We want to make a batch of size (fov*fov, nb_internal_states) by repeating the internal states
        internal_states = internal_states.repeat(fov*fov, 1)

        # Now, we add the internal states to each pixel to have a batch of size (fov*fov, nb_world_channels + nb_internal_states)
        x = torch.cat((x, internal_states), dim=1)

        # We pass the batch through the MLP
        x = self.MLP(x)

        # We reshape the output to have the same shape as the input
        x = x.view(-1, fov, fov)

        return x


class YaacBrainCNN(nn.Module):
    """
    This brain is a CNN that takes as input the square surrounding the Yaac.
    It is a basic CNN with 3 convolutional layers and 2 fully connected layers.
    For now, the output is an acceleration on the angle and norm of the speed vector.
    Future versions may include more complex outputs.

    This brain is very elegant as it gives sense to... senses, like vision, smell, pheromones, etc.
    However, it will be very difficult to include hunger, pain, etc. in the input. The other brains may be better for that.
    """
    def __init__(self, world: World, nb_internal_states):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv2d(world.nb_world_channels, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU()
        )

        self.MLP = YaacBrainMLP(world, nb_internal_states, 16)
    
    def forward(self, input, internal_states):
        # input has shape (nb_world_channels, fov, fov)
        # output has shape (nb_possible_actions)
        x = input.unsqueeze(0)
        x = self.CNN(x).squeeze(0)
        x = self.MLP(x, internal_states)
        return x

BRAIN_DICT = {
    'YaacBrainVector': YaacBrainVector,
    'YaacBrainMLP': YaacBrainMLP,
    'YaacBrainCNN': YaacBrainCNN
}

def get_random_color():
    # color : RGB + IR + UV
    return torch.randn(5).to(DEVICE)

def get_circle(color):
    circle = torch.zeros((len(RGB_CHANNELS + IR_UV_CHANNELS), 2*OBJECT_RADIUS, 2*OBJECT_RADIUS)).to(DEVICE)
    for i in range(2*OBJECT_RADIUS):
        for j in range(2*OBJECT_RADIUS):
            if (i-OBJECT_RADIUS)**2 + (j-OBJECT_RADIUS)**2 <= OBJECT_RADIUS**2:
                circle[:, i, j] = color
    
    return circle

def get_random_pheromones():
    return torch.randn(len(PHEROMONE_CHANNELS), 2*OBJECT_RADIUS, 2*OBJECT_RADIUS).to(DEVICE)

class YaacGenome:
    # The genome may include other parameters in the future, like brain architecture.
    # For now, only the field of view is variable and the brain has a fixed architecture.
    # This is easier to implement and to cross over.
    def __init__(self,
            world: World,
            # Field of View is in pixel. It measures the side of the square viewed by the Yaac.
            fov=31,
            max_health=100,
            max_energy=100,
            max_speed=2.,
            nb_internal_states=8,
            color=None
    ):
        self.fov = fov
        self.max_health = max_health
        self.max_energy = max_energy
        self.max_speed = max_speed

        self.pheromones = get_random_pheromones()
        self.color = color if color is not None else get_random_color()
        self.physical_representation = get_circle(self.color)

        self.brain = BRAIN_DICT[DEFAULT_BRAIN](world, nb_internal_states).to(DEVICE)

# Yet Another Artificial Creature.
# The Yaac is the base class for all creatures in the game.
class Yaac(MovingObject):
    def __init__(self,
            world: World,
            x=WIDTH//2,
            y=HEIGTH//2,
            speed=0,
            direction=0
        ):
        super().__init__(x, y, speed, direction)
        self.world = world

        self.stats = YaacStats()
        self.nb_internal_states = 8

        self.genome = YaacGenome(world, nb_internal_states=self.nb_internal_states)

        top, bottom, left, right = self.get_tb_lr()
        self.world.world_tensor[RGB_CHANNELS+IR_UV_CHANNELS, top:bottom, left:right] += self.genome.physical_representation
    
    def get_internal_state(self):
        return torch.tensor([
            self.stats.health,
            self.stats.energy,
            self.stats.hunger,
            self.stats.max_health,
            self.stats.max_energy,
            self.stats.max_hunger,
            self.stats.max_speed,
            self.stats.age
        ])

    def get_brain_output(self):
        view = self.world.get_fov(self.x, self.y, self.genome.fov)
        internal_state = self.get_internal_state()

        return self.genome.brain(view, internal_state)
    
    def get_new_speed(self, speed_channel):
        # TODO : the speed should have a constant norm and each pixel of the grid should be normalised.
        # Otherwize, pixels far from the Yacc will be naturally more important than the ones close to it.

        output_size = speed_channel.shape[1]
        speed_channel = speed_channel.reshape(-1)
        speed_weight = torch.softmax(speed_channel, dim=0)
        speed_weight = speed_weight.reshape(output_size, output_size)

        #get the grid of coordinates centered on the Yaac
        grid = torch.meshgrid(torch.arange(output_size).to(DEVICE), torch.arange(output_size).to(DEVICE), indexing='ij')
        grid = torch.stack(grid)
        grid[0] = grid[0].flip(0)
        grid -= self.genome.fov // 2

        grid = grid.flip(0)
        grid = grid.permute(1, 2, 0) * 1.
        
        #new speed vector is the weighted sum of the grid
        new_speed = torch.einsum('ijk,ij->k', grid, speed_weight)

        return torch.tensor([1., 1.])
    
    def big_brain_time(self):
        brain_output = self.get_brain_output()
        
        # get the new speed and direction from the first channel of the brain output
        speed_channel = brain_output[0]
        new_speed = self.get_new_speed(speed_channel)
        self.speed = min(new_speed.norm().item(), self.genome.max_speed)
        new_speed = new_speed[0] + 1j * new_speed[1]
        self.direction = new_speed.angle().item()

        # TODO : get other actions from the other channels

    def get_tb_lr(self):
        top = int(self.y - OBJECT_RADIUS)
        bottom = int(self.y + OBJECT_RADIUS)
        left = int(self.x - OBJECT_RADIUS)
        right = int(self.x + OBJECT_RADIUS)
        return top, bottom, left, right
    
    def release_pheromones(self):
        top, bottom, left, right = self.get_tb_lr()
        self.world.world_tensor[PHEROMONE_CHANNELS, top:bottom, left:right] += self.genome.pheromones
    
    def world_move(self):
        top, bottom, left, right = self.get_tb_lr()
        self.world.world_tensor[RGB_CHANNELS+IR_UV_CHANNELS, top:bottom, left:right] -= self.genome.physical_representation

        self.move()

        top, bottom, left, right = self.get_tb_lr()
        self.world.world_tensor[RGB_CHANNELS+IR_UV_CHANNELS, top:bottom, left:right] += self.genome.physical_representation
    
    def update(self):
        self.big_brain_time()
        #self.world_move()
        #self.release_pheromones()
        self.stats.age += 1
        self.stats.hunger += 1
        # TODO : energy cost depend on many things, like speed, size, etc.
        self.stats.energy -= 1