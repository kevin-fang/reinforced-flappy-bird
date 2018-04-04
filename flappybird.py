#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import os
import random

from neural_model import get_jump

global SAVING
SAVING = False

def makeDirIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class FlappyBird:
    def __init__(self):
        # set up the display
        self.screen = pygame.display.set_mode((400, 708))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha()]
        # create pipes
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()

        # set the gap between the pipes
        self.gap = 130

        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 8
        self.dead = False
        self.sprite = 0

        self.offset = random.randint(-110, 110)

        # point counter
        self.counter = 0

        # counters for image storage
        self.image_counter = 0
        self.game_counter = 0

        # make the first screenshot folder
        if SAVING:
        	makeDirIfNotExist("./screenshots")
        	makeDirIfNotExist("./screenshots/game{}/".format(self.game_counter))

    # move the walls to the left or teleport them to the end
    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)

    def birdUpdate(self):
        # if jumping, account for acceleration and lower the bird
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            # move the bird down
            self.birdY += self.gravity
            self.gravity += 0.2


        self.bird[1] = self.birdY

        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())

        # if collide with up pipe
        if upRect.colliderect(self.bird):
            self.dead = True
            # make the game reset immediately
            self.bird[1] = -1

        # if collide with the down pipe
        if downRect.colliderect(self.bird):
            self.dead = True
            # make the game reset immediately
            self.bird[1] = -1

        # check if bird has fallen above/below the screen
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            # reset the image counter and increment the game counter
            self.game_counter += 1
            self.image_counter = 0
            if SAVING:
            	makeDirIfNotExist("./screenshots/game{}/".format(self.game_counter))

    def run(self):
        # initialize game and game counter font
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)

        while True:
            clock.tick(60)
            event = pygame.event.wait()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10

            self.screen.fill((255, 255, 255))
            # keep this line commented out for training - less distracting background

            # self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (0, 0, 0)),
                                (200, 50))
            
            # change sprite 
            if self.jump:
                self.sprite = 1

            self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
            if not self.dead:
                self.sprite = 0
            
            self.updateWalls()
            self.birdUpdate()
            if SAVING:
            	pygame.image.save(self.screen, "screenshots/game{}/screenshot{}.jpg".format(self.game_counter, self.image_counter))
            self.image_counter += 1

            pygame.display.update()

if __name__ == "__main__":
    FlappyBird().run()
