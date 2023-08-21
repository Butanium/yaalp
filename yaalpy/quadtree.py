"""
Implementation of a quadtree data structure.
A quadtree is a spatial tree with the following properties:
    - Each node represents a square region of space.
    - Each node has a maximum capacity of points.
    - If a node exceeds its capacity, it splits into four subnodes.
"""
import numpy as np

from constants import QUADTREE_MAX_CAPACITY

def distance(p1, p2):
    """
    Return the distance between two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class QuadTree:
    def __init__(self, top_left, height, width, capacity=QUADTREE_MAX_CAPACITY):
        self.top_left = top_left
        self.height = height
        self.width = width

        # If false, this node is a leaf. Otherwise, it has four children.
        self.divided = False

        self.capacity = capacity
        self.points = []

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None
    
    def total_points(self):
        """
        Return the total number of points in this quadtree.
        """
        if not self.divided:
            return len(self.points)
        else:
            return (self.NE.total_points() + self.NW.total_points() +
                    self.SE.total_points() + self.SW.total_points())
    
    def contains(self, point):
        """
        Return true if this quadtree contains the given point.
        """
        x = self.top_left[0]
        y = self.top_left[1]
        px = point.x
        py = point.y
        return (px >= x and px < x + self.width and
                py >= y and py < y + self.height)

    def subdivide(self):
        """
        Subdivide this quadtree into four subnodes.
        """
        x = self.top_left[0]
        y = self.top_left[1]
        h = self.height / 2
        w = self.width / 2

        self.NE = QuadTree((x + w, y), h, w, self.capacity)
        self.NW = QuadTree((x, y), h, w, self.capacity)
        self.SE = QuadTree((x + w, y + h), h, w, self.capacity)
        self.SW = QuadTree((x, y + h), h, w, self.capacity)

        for point in self.points:
            self.NE.insert(point)
            self.NW.insert(point)
            self.SE.insert(point)
            self.SW.insert(point)

        self.divided = True
        
    def total_leafs(self):
        """
        Return the total number of leafs in this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return (self.NE.total_leafs() + self.NW.total_leafs() +
                    self.SE.total_leafs() + self.SW.total_leafs())
    
    def merge(self):
        """
        Try to merge the children of this quadtree.
        """
        if not self.divided:
            return
        #counting all points every time is very costly, so we count only when all four children are leaves
        if self.NE.divided or self.NW.divided or self.SE.divided or self.SW.divided:
            return
        if self.total_points() > self.capacity:
            return
        
        # theoretically, there is no need to recursively merge children as they are merged in the supress method called before the current merge
        self.points = []
        self.divided = False

        for point in self.NE.points:
            self.points.append(point)
        for point in self.NW.points:
            self.points.append(point)
        for point in self.SE.points:
            self.points.append(point)
        for point in self.SW.points:
            self.points.append(point)

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None
    
    def insert(self, point):
        if not self.contains(point):
            return
        
        if self.divided:
            self.NE.insert(point)
            self.NW.insert(point)
            self.SE.insert(point)
            self.SW.insert(point)
        else:
            self.points.append(point)
            if len(self.points) > self.capacity:
                self.subdivide()

    def supress(self, point):
        if not self.contains(point):
            return
        
        if self.divided:
            self.NE.supress(point)
            self.NW.supress(point)
            self.SE.supress(point)
            self.SW.supress(point)

            self.merge()
        else:
            self.points.remove(point)

    def closest(self, x, y):
        """
        Return the closest point to the given point.
        TODO
        """
        pass

    def query_rect(self, top_left, height, width):
        """
        Return all points in the given rectangle.
        """
        points = []
        if not self.intersects(top_left, height, width):
            return points

        if not self.divided:
            for point in self.points:
                if point.x >= top_left[0] and point.x < top_left[0] + width and point.y >= top_left[1] and point.y < top_left[1] + height:
                    points.append(point)
        else:
            points += self.NE.query_rect(top_left, height, width)
            points += self.NW.query_rect(top_left, height, width)
            points += self.SE.query_rect(top_left, height, width)
            points += self.SW.query_rect(top_left, height, width)

        return points

    def intersects(self, top_left, height, width):
        """
        Return true if the given rectangle intersects this quadtree.
        """
        x = self.top_left[0]
        y = self.top_left[1]
        return not (x > top_left[0] + width or x + self.width < top_left[0] or y > top_left[1] + height or y + self.height < top_left[1])
    
    def query_circle(self, center, radius):
        """
        Return all points in the given circle.
        """
        points = []
        if not self.intersects_circle(center, radius):
            return points

        if not self.divided:
            for point in self.points:
                if distance((point.x, point.y), center) <= radius:
                    points.append(point)
        else:
            points += self.NE.query_circle(center, radius)
            points += self.NW.query_circle(center, radius)
            points += self.SE.query_circle(center, radius)
            points += self.SW.query_circle(center, radius)

        return points
    
    def intersects_circle(self, center, radius):
        """
        Return true if the given circle intersects this quadtree.
        May also return true if the circle does not intersect this quadtree, we in fact check if the square containing the circle intersects the quadtree.
        """
        x = self.top_left[0]
        y = self.top_left[1]
        return not (x > center[0] + radius or x + self.width < center[0] - radius or y > center[1] + radius or y + self.height < center[1] - radius)