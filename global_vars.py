import pygame
from pygame.locals import *

global JUMP_CONST
global STAY_CONST
global JUMP
global STAY

JUMP_CONST = USEREVENT + 1
STAY_CONST = USEREVENT + 2
JUMP = pygame.event.Event(JUMP_CONST)
STAY = pygame.event.Event(STAY_CONST)