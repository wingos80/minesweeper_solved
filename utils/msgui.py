# msgui.py

import pygame as pg
from conf import *

class NumberDisplay:
    """ A class to show numbers in a display look """

    def __init__(self, initial_value):
        self.numbers = pg.image.load(NUMBERS_IMG_SOURCE)
        self.surf = surf = pg.Surface((39, 23))
        self.value = None
        self.set_value(initial_value)

    def set_value(self, value):
        """ Blits the corresponding images in the surface according to
        the given value """

        if value == self.value:
            return

        str_number = str(value).zfill(3)

        for i, n in enumerate(str_number):
            if n == '-':
                img_index = 10
            else:
                img_index = int(n)

            self.surf.blit(
                self.numbers.subsurface((13*img_index, 0, 13, 23)),
                (i*13, 0)
            )

        self.value = value


class SmileButton:
    """ The centered smiling button to display the state of the game and
    to restart the game when pressed """

    def __init__(self, app, pos, cmd=None, cmd_args=()):
        # Pointer to the main app class
        self.app = app

        self.pos = pos
        self.size = (26, 26)
        self.cmd = cmd
        self.cmd_args = cmd_args

        self.buttons = self.load_buttons()
        self.button_index = 1

        self.pressing = False

    def load_buttons(self):
        """ Loads the smile button states """

        faces = pg.image.load(BUTTONFACES_IMG_SOURCE)
        buttons = list()

        for i in range(5):
            buttons.append(
                faces.subsurface((26*i, 0, 26, 26))
            )

        return buttons

    def draw(self, screen):
        """ Draws the button on the given screen surface """

        mx, my = pg.mouse.get_pos()
        x, y = self.pos
        w, h = self.size

        hovering = (x < mx < x+w) and (y < my < y+h)
        clicking = pg.mouse.get_pressed()[0]
        pressing = hovering and clicking

        if not self.pressing and pressing:
            # Button pressed
            self.pressing = True

        if not clicking and self.pressing:
            # Button released
            self.pressing = False

            # If still hovering and there is a command, perform it
            if hovering and self.cmd is not None:
                self.cmd(*self.cmd_args)

        self.button_index = 1

        if clicking and self.app.alive and not self.app.won:
            self.button_index = 3

        if self.app.won:
            self.button_index = 0

        if not self.app.alive:
            self.button_index = 4

        if pressing:
            self.button_index = 2

        screen.blit(self.buttons[self.button_index], self.pos)
