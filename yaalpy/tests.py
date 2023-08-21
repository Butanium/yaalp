import numpy as np
import time

from creature import MovingObject
from quadtree import QuadTree

quad_tree = QuadTree((0, 0), 1024, 1024)

t1 = time.perf_counter()
nb_points = 1e4
points = [MovingObject(np.random.randint(0, 1024), np.random.randint(0, 1024), speed=np.random.rand(1)*1, direction=np.random.rand(1)*2*np.pi) for _ in range(int(nb_points))]
t2 = time.perf_counter()

print(f"Time to create {nb_points} points: {t2-t1} seconds")

t1 = time.perf_counter()
for point in points:
    quad_tree.insert(point)
t2 = time.perf_counter()

print(f"Time to insert {nb_points} points: {t2-t1} seconds")

t1 = time.perf_counter()

print(quad_tree.total_leafs())

t2 = time.perf_counter()

print(f"Time to count {nb_points} points: {t2-t1} seconds")


t1 = time.perf_counter()
for point in points:
    quad_tree.supress(point)
    point.move()
    quad_tree.insert(point)
t2 = time.perf_counter()

print(f"Time to move {nb_points} points inside the tree: {t2-t1} seconds")


t1 = time.perf_counter()
new_tree = QuadTree((0, 0), 1024, 1024)
for point in points:
    point.move()
    new_tree.insert(point)
t2 = time.perf_counter()

print(f"Time to move {nb_points} points in a new tree: {t2-t1} seconds")