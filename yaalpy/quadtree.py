"""
Implementation of a quadtree data structure.
A quadtree is a spatial tree with the following properties:
    - Each node represents a square region of space.
    - Each node has a maximum capacity of points.
    - If a node exceeds its capacity, it splits into four subnodes.
"""
import numpy as np

from constants import QUADTREE_MAX_CAPACITY, OBJECT_RADIUS

def distance(p1, p2):
    """
    Return the distance between two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def sqr_distance(p1, p2):
    """
    Return the distance between two points without taking the square root.
    """
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def distance_point_to_sqr(sqr, point, distance_fn=distance):
    """
    Return the distance between a point and a square.
    """
    x = sqr.top_left[0]
    y = sqr.top_left[1]
    px = point[0]
    py = point[1]
    if px < x:
        if py < y:
            return distance_fn((px, py), (x, y))
        elif py > y + sqr.height:
            return distance_fn((px, py), (x, y + sqr.height))
        else:
            return (x - px)**2 if distance_fn == sqr_distance else x - px
    elif px > x + sqr.width:
        if py < y:
            return distance_fn((px, py), (x + sqr.width, y))
        elif py > y + sqr.height:
            return distance_fn((px, py), (x + sqr.width, y + sqr.height))
        else:
            return (px - x) ** 2 if distance_fn == sqr_distance else px - x
    else:
        if py < y:
            return (y - py) ** 2 if distance_fn == sqr_distance else y - py
        elif py > y + sqr.height:
            return (py - y + sqr.height) ** 2 if distance_fn == sqr_distance else py - y + sqr.height
        else:
            return 0

class QuadTree:
    def __init__(self, top_left, height, width, capacity=QUADTREE_MAX_CAPACITY):
        self.top_left = top_left
        self.height = height
        self.width = width

        # If false, this node is a leaf. Otherwise, it has four children.
        self.divided = False

        self.capacity = capacity
        self.points = []
        self.representative = None
        self.empty = True

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None

    def depth(self):
        """
        Return the depth of this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return 1 + max(self.NE.depth(), self.NW.depth(), self.SE.depth(), self.SW.depth())
    
    def total_points(self):
        """
        Return the total number of points in this quadtree.
        """
        if not self.divided:
            return len(self.points)
        else:
            return (self.NE.total_points() + self.NW.total_points() +
                    self.SE.total_points() + self.SW.total_points())
        
    def total_leafs(self):
        """
        Return the total number of leafs in this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return (self.NE.total_leafs() + self.NW.total_leafs() +
                    self.SE.total_leafs() + self.SW.total_leafs())
    
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
        
        self.representative = point
        if self.divided:
            self.NE.insert(point)
            self.NW.insert(point)
            self.SE.insert(point)
            self.SW.insert(point)
        else:
            self.points.append(point)
            if len(self.points) > self.capacity:
                self.subdivide()
        
        self.empty = False

    # TODO : This method doesn't work !!!
    # We often want to supress points that moved outside of their previous square.
    def supress(self, point):
        if not self.contains(point):
            return
        
        if point == self.representative :
            self.representative = None

        if self.divided:
            self.NE.supress(point)
            self.NW.supress(point)
            self.SE.supress(point)
            self.SW.supress(point)

            self.representative = self.NE.representative or self.NW.representative or self.SE.representative or self.SW.representative

            self.merge()
            if self.total_points() == 0:
                self.empty = True
        else:
            self.points.remove(point)
            if len(self.points) > 0:
                self.representative = self.points[0]
            else:
                self.empty = True
    
    def naive_closest(self, point, radius=OBJECT_RADIUS+1):
        """
        This has a worst case complexity of O(n*log(n)) for one query. It is worse thant the naive method of going through the list of all points.
        However, the worst case complexity assumes that all points are within the same radius, so all points are mutually in collision.
        Thus, in a setting with collisions, it is extremely unlikely that more than 6 points are close enough, and in practice, this
        method is faster than the mathematically optimal method below.
        """
        candidates = self.query_circle((point.x, point.y), radius)
        return min(candidates, key=lambda p: sqr_distance((p.x, p.y), (point.x, point.y)))

    def closest(self, x, y):
        """
        Return the closest point to the given point.
        Algorithm :
            Maintain a list of interesting squares
            sqrs_0 is the root of the quadtree
            sqrs_{i+1} are the squares whose distance to the point is less than the current minimum distance among the children of sqrs_i.
            The current minimum distance is the distance between the point and the best representative yet.
            Returning the best representative gives the closest point.
        """
        if not self.divided:
            # This case is juste for trees reduced to a single leaf. Plunging into the tree will exclude empty squares.
            if len(self.points) == 0:
                return None
            return min(self.points, key=lambda p: sqr_distance((p.x, p.y), (x, y)))
        
        sqrs = [self]
        
        while sqrs != []:
            # /!\ only put non empty squares in sqrs
            # change here to include capacity > 1 trees
            best = min([sqr.representative for sqr in sqrs],
                           key=lambda p: sqr_distance((p.x, p.y), (x, y))
                          )
            
            new_sqrs = []
            for sqr in sqrs:
                if not sqr.divided :
                    # In this case, the closest point in sqr is already accounted for in best_rep and we can skip to the next square.
                    continue
                
                for sub_sqr in [sqr.NE, sqr.NW, sqr.SE, sqr.SW]:
                    if (not sub_sqr.empty) \
                        and (distance_point_to_sqr(sub_sqr, (x, y), distance_fn=sqr_distance) < sqr_distance((best.x, best.y), (x, y))):
                        new_sqrs.append(sub_sqr)

                        if not sub_sqr.divided:
                            # the representative is changed to the closest :
                            sub_sqr.representative = min(sub_sqr.points, key=lambda p: sqr_distance((p.x, p.y), (x, y)))
            
            sqrs = new_sqrs

        return best

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

        # this is much slower in practice than the above method
        dist = distance_point_to_sqr(self, center)
        return dist <= radius**2