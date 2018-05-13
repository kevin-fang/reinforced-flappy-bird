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

global FRAME_SCORE
global PIPE_SCORE
global DEATH_SCORE

FRAME_SCORE = .01
PIPE_SCORE = 1
DEATH_SCORE = -100