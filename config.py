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
STARTING_NUM = 1000

# score threshold needed to save game
global SCORE_THRESHOLD
SCORE_THRESHOLD = 200

# how many games to save above the threshold
global NUM_SAVES
NUM_SAVES = 500

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