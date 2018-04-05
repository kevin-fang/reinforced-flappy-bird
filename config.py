import pygame
from pygame.locals import *  # noqa

# save screenshots of the game
global SAVING
SAVING = True

# run headless - for servers
global HEADLESS
HEADLESS = False

# where to start counting for game numbers - so we don't overwrite data
global STARTING_NUM
STARTING_NUM = 0

global SAVE_THRESHOLD
SAVE_THRESHOLD = 150

global DATA_DIR
DATA_DIR = './data/'

global MODEL_DIR
MODEL_DIR = './models'