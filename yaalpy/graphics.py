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