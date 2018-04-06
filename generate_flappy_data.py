#!/usr/bin/env python

import pygame, sys, os, random
from pygame.locals import *  # noqa
from random_jumper import get_jump
from config import *
from global_vars import *
from shutil import rmtree

def makeDirIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class FlappyBird:
    def __init__(self):
        # set up the display
        if HEADLESS:
           # os.environ["SDL_VIDEODRIVER"] = "dummy"
            #self.real_screen = pygame.display.set_mode((1,1))
            self.real_screen = pygame.display.set_mode((1, 1))
            self.screen = pygame.Surface((400, 708)).convert()
        else:
            self.screen = pygame.display.set_mode((400, 708))

        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert(),
                            pygame.image.load("assets/2.png").convert()]
        # create pipes
        self.wallUp = pygame.image.load("assets/bottom.png").convert()
        self.wallDown = pygame.image.load("assets/top.png").convert()

        # set the gap between the pipes
        self.gap = 250

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

        # counts since last jump
        self.last_jump_counter = -1
        # frames since starting
        self.alive_frames = 0

        # counters for image storage
        self.image_counter = 0
        self.game_counter = STARTING_NUM

        self.saved_game_counter = 0

        # make the first screenshot folder
        if SAVING:
            makeDirIfNotExist(TRAIN_SCREEN_DIR)
            makeDirIfNotExist(os.path.join(TRAIN_SCREEN_DIR, "game{}".format(self.game_counter)))

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

        # up and down pipes
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())

        # if collide with pipe
        if upRect.colliderect(self.bird) or downRect.colliderect(self.bird):
            self.dead = True
            # make the game reset immediately
            self.bird[1] = -1

        # check if bird has fallen above/below the screen
        if not 0 < self.bird[1] < 720:
            screens_folder = os.path.join(TRAIN_SCREEN_DIR, "game{}".format(self.game_counter))
            if SAVING:
                if self.saved_game_counter >= NUM_SAVES:
                    print("Successfully generated: {} games past {} frame threshold.".format(NUM_SAVES, SCORE_THRESHOLD))
                    sys.exit()
                if self.alive_frames > SCORE_THRESHOLD:
                    self.saved_game_counter += 1
                    print("Keeping game: {}; alive frames: {}. Saved so far: {}".format(self.game_counter, self.alive_frames, self.saved_game_counter))
                else:
                    print("Deleting game: {}; alive frames: {}".format(self.game_counter, self.alive_frames))
                    rmtree(screens_folder)
            self.alive_frames = 0
            self.bird[1] = 350
            self.birdY = 350
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            # reset the image counter and increment the game counter
            self.game_counter += 1
            self.image_counter = 0
            if SAVING:
                makeDirIfNotExist(os.path.join(TRAIN_SCREEN_DIR, "game{}".format(self.game_counter)))

    def frameUpdate(self, jump):
        self.screen.fill((255, 255, 255))
        # keep this line commented out for training - less distracting background
        #self.screen.blit(self.background, (0, 0))

        self.screen.blit(self.wallUp,
                         (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                         (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(self.font.render(str(self.counter),
                                    0,
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
        if (SAVING and self.alive_frames % 3 == 0 and jump != None) or self.last_jump_counter == 0:
            pygame.image.save(self.screen, os.path.join(TRAIN_SCREEN_DIR, "game{}".format(self.game_counter), 
                                                    "{}_{}_screenshot_{}.jpg".format("j" if jump else "n", self.last_jump_counter, self.image_counter)))
        self.image_counter += 1

        pygame.display.update()

    def run(self):
        # initialize game and game counter font
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        self.font = font
        self.frameUpdate(jump = None)

        while True:
            clock.tick()
            if get_jump(None, self.last_jump_counter):
                self.last_jump_counter = 0
                pygame.event.post(JUMP)
            else:
                self.last_jump_counter += 1
                pygame.event.post(STAY)

            event = pygame.event.wait()
            #for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == JUMP_CONST and not self.dead:
                self.jump = 17
                self.gravity = 5
                self.jumpSpeed = 10
                self.alive_frames += 1
                self.frameUpdate(jump = True)
            elif event.type == STAY_CONST and not self.dead:
                #print(self.last_jump_counter)
                self.alive_frames += 1
                self.frameUpdate(jump = False)                

if __name__ == "__main__":
    FlappyBird().run()
