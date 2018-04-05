import pygame
from pygame.locals import *  # noqa

# save screenshots of the game
global SAVING
SAVING = True

# run headless - for servers
global HEADLESS
HEADLESS = True

# where to start counting for game numbers - so we don't overwrite data
global STARTING_NUM
STARTING_NUM = 186