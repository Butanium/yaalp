import pygame as pg
import time
import numpy as np
import torch

import graphics
from world import World
import creature

screen = graphics.get_screen()

with torch.no_grad():
    world = World()
    yaacs = [creature.Yaac(world, 100, 100) for i in range(1000)]

    while True:
        t1 = time.perf_counter()
        events = graphics.get_events()
        if pg.QUIT in [event.type for event in events]:
            print("quit")
            graphics.quit()
            break

        world.update()
        for yaac in yaacs:
            yaac.update()

        graphics.draw_world(world, screen)

        t2 = time.perf_counter()
        print("time :", (t2-t1), "s")