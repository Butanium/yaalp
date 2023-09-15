import numpy as np
import time
import pygame as pg

import graphics
from quadtree import QuadTree, sqr_distance

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def naive_closest(point, points):
    candidates = points.copy()
    candidates.remove(point)
    return min(candidates, key=lambda p: sqr_distance((p.x, p.y), (point.x, point.y)))

screen = graphics.get_screen()

tree_size = 1024
quad_tree = QuadTree((0, 0), tree_size, tree_size, draw=True, screen=screen)
nb_points = 1e3
points = [Point(np.random.randint(0, tree_size), np.random.randint(0, tree_size)) for _ in range(int(nb_points))]

duplicate = []
for point in points:
    closest = naive_closest(point, points)
    if sqr_distance((point.x, point.y), (closest.x, closest.y)) == 0:
        duplicate.append(point)
for dup in duplicate:
    points.remove(dup)
for point in points:
    closest = naive_closest(point, points)
    assert sqr_distance((point.x, point.y), (closest.x, closest.y)) != 0, f"Point {point.x, point.y} is not unique"
nb_points = len(points)

for point in points:
    quad_tree.insert(point)

# draw the quadtree : draw a rectangle of thickness 1 for each node
# draw the points : draw a circle of radius 3 for each point

graphics.draw_quadtree(screen, quad_tree)
graphics.update_screen()

while True:
    for event in graphics.get_events():
        if event.type == pg.QUIT:
            graphics.quit()
            exit()
        # if clic somewhere, create a point there, draw it in red, and find its closest neighbor with animation
        if event.type == pg.MOUSEBUTTONUP:
            graphics.draw_quadtree(screen, quad_tree)
            x, y = pg.mouse.get_pos()
            point = Point(x, y)
            graphics.draw_point(screen, point, color=(255, 0, 0))
            graphics.update_screen()
            closest = quad_tree.closest_depth(point, order_regions=True)
    
    time.sleep(0.1)
