import pygame
import random
from sprites import *

global PLAY_SOUNDS
PLAY_SOUNDS = False

global FPS
FPS = 300

global SCREENWIDTH
SCREENWIDTH  = 288

global SCREENHEIGHT
SCREENHEIGHT = 512

global PIPEGAPSIZE
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 150 # gap between upper and lower part of pipe

global BASEY
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
global IMAGES
global SOUNDS
global HITMASKS
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
