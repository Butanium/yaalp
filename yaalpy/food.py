"""
Food might take several forms:
-evolving plants
-fixed carrion

all with radius "0" for collision


As they are supposedly immobile, they will be in a different quadtree than the moving objects.
The quadtree of moving objects is reconstructed at each frame, while the quadtree of food is
only updated when a food is eaten or a new food is created.
"""

from constants import WIDTH, HEIGTH, DEFAULT_WORLD_IMPRINT_FOOD, RGB_CHANNELS, IR_UV_CHANNELS, PHEROMONE_CHANNELS, DEVICE

class Food:
    def __init__(self,  world, x, y,energy=10):
        self.x = x
        self.y = y
        self.energy = energy

        self.world = world
        self.world_imprint = DEFAULT_WORLD_IMPRINT_FOOD
        self.first_imprint()
    
    def first_imprint(self, negative=False):
        top, bottom, left, right = self.get_bounding()
        print(self.world.world_tensor[PHEROMONE_CHANNELS, top:bottom, left:right].shape)
        print(self.world_imprint[PHEROMONE_CHANNELS].shape)
        self.world.world_tensor[:, top:bottom, left:right] += self.world_imprint * (-1 if negative else 1)

    def get_bounding(self):
        top = max(0, self.y - 2)
        bottom = min(HEIGTH - 1, self.y + 3)
        left = max(0, self.x - 2)
        right = min(WIDTH - 1, self.x + 3)

        return top, bottom, left, right
    
    def update(self):
        # TODO : Upon searching for optimisations, try to make food pheromone outside of the evaporation process to make this function useless.
        top, bottom, left, right = self.get_bounding()
        self.world.world_tensor[PHEROMONE_CHANNELS, top:bottom, left:right] += self.world_imprint[PHEROMONE_CHANNELS]
    
    def die(self):
        self.first_imprint(negative=True)

class PlantGenome:
    def __init__(self, energy=10):
        self.energy = energy

        self.world_imprint = DEFAULT_WORLD_IMPRINT_FOOD
    
    def mutate(self):
        # TODO
        pass

class Plant(Food):
    def __init__(self, x, y, world, energy=10):
        super().__init__(x, y, world, energy)

        self.genome = PlantGenome(energy)
        self.first_imprint(negative=True)
        self.world_imprint = self.genome.world_imprint
        self.first_imprint(negative=False)
