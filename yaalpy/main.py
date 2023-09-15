import pygame as pg
import time
import numpy as np
import torch

import graphics
from world import World
import creature
from food import Food
from quadtree import QuadTree
from constants import HEIGTH, WIDTH

quad_food = QuadTree((0, 0), WIDTH, HEIGTH)
quad_yaac = QuadTree((0, 0), WIDTH, HEIGTH)

screen = graphics.get_screen()

nb_yaacs = 100
nb_food = 100

with torch.no_grad():
    t1 = time.perf_counter()
    world = World()
    t2 = time.perf_counter()
    print("world init :", round((t2-t1)*1000, 2), "ms")
    t1 = time.perf_counter()
    yaacs = [creature.Yaac(world, np.random.randint(10, WIDTH-10), np.random.randint(10, HEIGTH-10)) for i in range(nb_yaacs)]
    t2 = time.perf_counter()
    print("yaac init :", round((t2-t1)*1000, 2), "ms")
    t1 = time.perf_counter()
    foods = [Food(world, np.random.randint(10, WIDTH-10), np.random.randint(10, WIDTH-10)) for i in range(nb_food)]
    t2 = time.perf_counter()
    print("food init :", round((t2-t1)*1000, 2), "ms")

    t1 = time.perf_counter()
    for yaac in yaacs:
        quad_yaac.insert(yaac)
    for food in foods:
        quad_food.insert(food)
    t2 = time.perf_counter()
    print("quadtree insert :", round((t2-t1)*1000, 2), "ms")

    while True:
        t0 = time.perf_counter()
        events = graphics.get_events()
        if pg.QUIT in [event.type for event in events]:
            print("quit")
            graphics.quit()
            break

        t1 = time.perf_counter()
        world.update()
        t2 = time.perf_counter()
        print("world update :", round((t2-t1)*1000, 2), "ms")

        # TODO : world update should also update everything else.
        # world should have the list of all objects, the quadtree, take care of collisions, etc.
        t1 = time.perf_counter()
        for yaac in yaacs:
            yaac.update()
        t2 = time.perf_counter()
        print("yaac update :", round((t2-t1)*1000, 2), "ms")

        t1 = time.perf_counter()
        for food in foods:
            food.update()
        t2 = time.perf_counter()
        print("food update :", round((t2-t1)*1000, 2), "ms")

        t1 = time.perf_counter()
        quad_yaac.clear()
        for yaac in yaacs:
            quad_yaac.insert(yaac)
        t2 = time.perf_counter()
        print("quadtree yaac update :", round((t2-t1)*1000, 2), "ms")

        t1 = time.perf_counter()
        for yaac in yaacs:
            closest_food = quad_food.closest_naive(yaac)
            closest_yaac = quad_yaac.closest_naive(yaac)
        t2 = time.perf_counter()
        print("collisions :", round((t2-t1)*1000, 2), "ms")

        t1 = time.perf_counter()
        graphics.draw_world(world, screen)
        t2 = time.perf_counter()
        print("draw world :", round((t2-t1)*1000, 2), "ms")

        t1 = time.perf_counter()
        print("total :", (t1-t0)*1000, "ms\n")