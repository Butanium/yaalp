import torch

from constants import HEIGTH, WIDTH, DEVICE, MAX_FOV, DEFAULT_NB_WORLD_CHANNELS, PHEROMONE_DIFFUSION_STRENGTH, PHEROMONE_EVAPORATION_FACTOR, PHEROMONE_CHANNELS

class World:
    def __init__(self) -> None:
        self.nb_world_channels = DEFAULT_NB_WORLD_CHANNELS
        self.height = HEIGTH
        self.width = WIDTH

        self.world_tensor = torch.zeros((self.nb_world_channels, self.height, self.width), device=DEVICE)
        self.padding = 0

        self.nb_possible_actions = 3
        
        #gaussian kernel for pheromone diffusion
        self.diffusion_device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        self.pheromone_diffusion_filter = self.get_pheromone_diffusion_filter()

    def get_pheromone_diffusion_filter(self):
        kernel = torch.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                kernel[i, j] = -((i-5//2)**2 + (j-5//2)**2)/(2*PHEROMONE_DIFFUSION_STRENGTH**2)
        kernel = torch.exp(kernel)
        kernel = kernel/torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(len(PHEROMONE_CHANNELS), 1, 1, 1)
        filter = torch.nn.Conv2d(len(PHEROMONE_CHANNELS),
                                 len(PHEROMONE_CHANNELS),
                                 5,
                                 groups=len(PHEROMONE_CHANNELS),
                                 bias=False,
                                 padding=2
                            )
        
        filter.weight.data = kernel
        filter.weight.requires_grad = False
        filter = filter.to(self.diffusion_device)

        return filter

    def get_fov(self, x, y, fov):
        left = int(x-fov//2) + MAX_FOV
        clamped_left = max(left, 0)
        right = left + fov
        clamped_right = min(right, self.width)
        top = int(y-fov//2) + MAX_FOV
        clamped_top = max(top, 0)
        bottom = top + fov
        clamped_bottom = min(bottom, self.height)
        #TODO : check that x and y axis are correct everywhere
        view = torch.zeros((self.nb_world_channels, fov, fov), device=DEVICE)
        print(x, y)
        print(top, bottom, left, right)
        print(clamped_top, clamped_bottom, clamped_left, clamped_right)
        print(self.world_tensor[:, clamped_top:clamped_bottom, clamped_left:clamped_right].shape)
        print(view[:, clamped_top-top:clamped_bottom-top, clamped_left-left:clamped_right-left].shape)
        view[:, clamped_top-top:clamped_bottom-top, clamped_left-left:clamped_right-left] = self.world_tensor[:, clamped_top:clamped_bottom, clamped_left:clamped_right]

        return view

    def pheromone_evaporation(self):
        self.world_tensor[PHEROMONE_CHANNELS] *= PHEROMONE_EVAPORATION_FACTOR

    def pheromone_diffusion(self):
        self.world_tensor[PHEROMONE_CHANNELS] = self.pheromone_diffusion_filter(self.world_tensor[PHEROMONE_CHANNELS].unsqueeze(0)).squeeze(0)
        
    def update(self):
        self.world_tensor = self.world_tensor.to(self.diffusion_device)
        self.pheromone_evaporation()
        self.pheromone_diffusion()
        self.world_tensor = self.world_tensor.to(DEVICE)