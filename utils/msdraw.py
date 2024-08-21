# msdraw.py

import pygame as pg

from conf import *

def draw_border(surf, pos, size, thickness, background_color, top_border_color,
                bot_border_color, inner_color):
    """ Draws border with the style of the original game borders """

    x, y = pos
    w, h = size[0]-1, size[1]-1

    t = thickness

    pg.draw.rect(surf, background_color, (x, y, w+1, h+1))

    if t == 1:
        pg.draw.rect(surf, top_border_color, (x, y, w, h))
        pg.draw.rect(surf, bot_border_color, (x+1, y+1, w, h))
        pg.draw.rect(surf, inner_color, (x+1, y+1, w-1, h-1))
        return

    pg.draw.polygon(surf, top_border_color,
        [(x, y),
        (x, y+h-1),
        (x+t-1, y+h-t),
        (x+t-1, y+t-1),
        (x+w-t, y+t-1),
        (x+w-1, y)])

    pg.draw.polygon(surf, bot_border_color,
        [(x+w, y+h),
        (x+w, y+1),
        (x+w+1-t, y+t),
        (x+w+1-t, y+h+1-t),
        (x+t, y+h+1-t),
        (x+1, y+h)])

    pg.draw.rect(surf, inner_color, (x+t, y+t, (w+1)-2*t, (h+1)-2*t))


def swap_color(surface, color_a, color_b):
    """ Changes all pixels of the surface from color A to color B,
    preserve transparency. """

    w, h = surface.get_size()
    r, g, b = color_b

    for x in range(w):
        for y in range(h):
            color_b = surface.get_at((x, y))
            if color_a[:3] == color_b[:3]:
                surface.set_at((x, y), pg.Color(r, g, b, color_b[3]))

    return surface


def render_cell(base_surf, symbol_pos, swap_colors=None):
    """ Returns a new surface pasting a symbol on base_surf """

    symbols_img = pg.image.load(SYMBOLS_IMG_SOURCE).convert_alpha()

    base_surf = base_surf.copy()
    symbol_surf = symbols_img.subsurface((symbol_pos * 16, 0, 16, 16))

    if swap_colors is not None:
        swap_color(symbol_surf, *swap_colors)

    symbol_surf = pg.transform.scale(
        symbol_surf, (CELL_SIZE, CELL_SIZE))
    base_surf.blit(symbol_surf, (0, 0))

    return base_surf
