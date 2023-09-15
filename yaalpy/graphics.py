import pygame as pg

from constants import HEIGTH, WIDTH, RGB_CHANNELS

def get_screen(width=WIDTH, heigth=HEIGTH):
    return pg.display.set_mode((width, heigth))

def fill_screen(screen, color=(0, 0, 0)):
    screen.fill(color)

def draw(drawables, screen):
    for drawable in drawables:
        drawable.draw(screen)

def update_screen():
    pg.display.flip()

def draw_loop(drawables, screen):
    fill_screen(screen)
    draw(drawables, screen)
    update_screen()

def draw_world(world, screen):
    rgb = world.world_tensor[RGB_CHANNELS].permute(2, 1, 0).cpu()
    rgb = (rgb.clamp(0, 1) * 255).numpy().astype(int)
    pg.surfarray.blit_array(screen, rgb)
    update_screen()

def quit():
    pg.quit()

def get_events():
    return pg.event.get()

def get_pressed():
    return pg.key.get_pressed()

def get_mouse_pos():
    return pg.mouse.get_pos()

def draw_rect(screen, x, y, w, h, thickness=1, color=(255, 255, 255)):
    pg.draw.rect(screen, color, (x, y, w, h), thickness)

def draw_circle(screen, x, y, radius, color=(255, 255, 255)):
    pg.draw.circle(screen, color, (x, y), radius)

class Drawable:
    def __init__(self):
        pass
    
    def draw(self, screen):
        pass

class Sprite(Drawable):
    def __init__(self, x, y, sprite):
        self.x = x
        self.y = y
        self.sprite = sprite
    
    def draw(self, screen):
        screen.blit(self.sprite, (self.x, self.y))

class Circle(Drawable):
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
    
    def draw(self, screen):
        pg.draw.circle(screen, self.color, (self.x, self.y), self.radius)


def draw_tree_rect(screen, tree, color=(255, 255, 255)):
    x, y = tree.top_left
    w, h = tree.width, tree.height
    draw_rect(screen, x, y, w, h, thickness=1, color=color)

def draw_point(screen, point, color=(255, 255, 255)):
    x, y = point.x, point.y
    draw_circle(screen, x, y, 1, color=color)

def draw_quadtree(screen, tree, color=(255, 255, 255)):
    draw_tree_rect(screen, tree, color)
    
    if tree.divided:
        draw_quadtree(screen, tree.NE, color)
        draw_quadtree(screen, tree.NW, color)
        draw_quadtree(screen, tree.SE, color)
        draw_quadtree(screen, tree.SW, color)

    else:
        for point in tree.points:
            draw_point(screen, point, color)