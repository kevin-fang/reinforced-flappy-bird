import pygame
from pygame.locals import *  # noqa

# run headless - for servers
global HEADLESS
HEADLESS = False

# where to start counting for game numbers - so we don't overwrite data
global STARTING_NUM
STARTING_NUM = 0

# directory to hold numpy arrays
global DATA_DIR
DATA_DIR = './data/'

# directory to hold TensorFlow models
global MODEL_DIR
MODEL_DIR = './models'

# directory to hold training screenshot data
global TRAIN_SCREEN_DIR
TRAIN_SCREEN_DIR = './screenshots'

# directory to hold testing screenshot data
global TEST_SCREEN_DIR
TEST_SCREEN_DIR = './screenshots_test'