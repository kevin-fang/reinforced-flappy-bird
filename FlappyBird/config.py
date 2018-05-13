import pygame
from pygame.locals import *  # noqa
import os

# run headless - for servers
global HEADLESS
HEADLESS = False

global NUM_NEURAL_DIMS
NUM_NEURAL_DIMS = 6

# leave the following untouched 

global CANVAS_HEIGHT
CANVAS_HEIGHT = 708

global CANVAS_WIDTH
CANVAS_WIDTH = 400

# used for image reading - deprecated, for now
global IMG_SCALE_FACTOR
IMG_SCALE_FACTOR = 0.2

# whether to save images to disk (needed if training)
global SAVING
SAVING = True

# log directory
global LOG_DIR
LOG_DIR = './log'

# name of model to save
global MODEL_NAME
MODEL_NAME = "trained_flappy"

# directory to hold TensorFlow models
global MODEL_DIR
MODEL_DIR = './models'

global MODEL_PATH
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# number of games in each training iteration
global NUM_GAMES
NUM_GAMES = 15

# directory to hold numpy arrays
global DATA_DIR
DATA_DIR = './data'

# directory to hold training screenshot data
global TRAIN_SCREEN_DIR
TRAIN_SCREEN_DIR = './screenshots'

# directory to hold testing screenshot data
global TEST_SCREEN_DIR
TEST_SCREEN_DIR = './screenshots_test'