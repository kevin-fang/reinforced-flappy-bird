from itertools import cycle
import random, sys, numpy as np, pygame
sys.path.append('../')
import neural_jumper

from pygame.locals import *
from config import *
from game_config import *
from global_vars import *
from sprites import *

global gameCounter
gameCounter = 0

try:
    xrange
except NameError:
    xrange = range

def initializeSprites():
    # select random background sprites
    #randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)

    randBg, randPlayer, pipeindex = 0, 0, 0
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    #randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    #pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

def initializeGame():
    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('./FlappyBird/assets/sprites/0.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/1.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/2.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/3.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/4.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/5.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/6.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/7.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/8.png').convert_alpha(),
        pygame.image.load('./FlappyBird/assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('./FlappyBird/assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('./FlappyBird/assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('./FlappyBird/assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('./FlappyBird/assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('./FlappyBird/assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('./FlappyBird/assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('./FlappyBird/assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('./FlappyBird/assets/audio/wing' + soundExt)

def main(model = False):
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')
    global gameCounter
    gameCounter = 0
    neural_jumper.initialize_network(model)

    initializeGame()
    initializeSprites()

    while True:
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        if crashInfo['over']:
            print("{} games played. Finishing iteration.".format(NUM_GAMES))
            pygame.quit()
            return
        #showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }


def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps
    flappedIteration = False
    lastJumpCounter = 0
    frameCounter = 0
    gameData = []
    actionData = []

    while True:

        data_arr = np.array([playerVelY,  playerRot, flappedIteration, playery, lastJumpCounter, *getPipeInfo(upperPipes), *getPipeInfo(lowerPipes)])
        #print(data_arr)

        global gameCounter
        
        frameCounter += 1

        flappedIteration = False

        result = neural_jumper.get_jump(data_arr)[0]

        if result == 1:
            pygame.event.post(JUMP)
        elif result == 0:
            pygame.event.post(STAY)

        event = pygame.event.wait()

        if event.type == QUIT:
            sys.exit()
        if event.type == JUMP_CONST:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                flappedIteration = True
                lastJumpCounter = 0
                if PLAY_SOUNDS: SOUNDS['wing'].play()
        if event.type == STAY_CONST:
            pass

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)

        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        pipeMidPos = upperPipes[0]['x'] + IMAGES['pipe'][0].get_width() / 2
        throughPipe = pipeMidPos <= playerMidPos < pipeMidPos + 4

        learning_score = -1
        if crashTest[0]: 
            learning_score = DEATH_SCORE
        elif throughPipe:
            learning_score = PIPE_SCORE
        else:
            learning_score = FRAME_SCORE

        screenshot_name = os.path.join(TRAIN_SCREEN_DIR, "game{}".format(gameCounter), 
                                                                    "{img_num}_{y}_{r}_{last_jump}_capture.npy"
                                                                    .format(y=1 if flappedIteration else 0, 
                                                                        r=learning_score, 
                                                                        last_jump=lastJumpCounter, 
                                                                        img_num=frameCounter))

        action = [frameCounter, flappedIteration, learning_score]

        gameData.append(data_arr)
        actionData.append(action)
        #np.save(screenshot_name, data_arr)
        # bird has crashed
        if crashTest[0]:

            lastJumpCounter = 0
            gameCounter += 1
            gameData = np.array(gameData)
            actionData = np.array(actionData)

            np.save(os.path.join(TRAIN_SCREEN_DIR, "game{}_data".format(gameCounter)), gameData)
            np.save(os.path.join(TRAIN_SCREEN_DIR, "game{}_action".format(gameCounter)), actionData)

            actionData = []
            gameData = []

            print("Game {} over; alive frames: {}".format(gameCounter, frameCounter))
            frameCounter = 0
            if gameCounter == NUM_GAMES:
                return {
                    'over': True
                }
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot,
                'over': False
            }

        # check for score
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if PLAY_SOUNDS: SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        else:
            lastJumpCounter += 1

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        if (score > 50):
            print("Score above 50. Quitting...")
            sys.exit(0)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        
        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getPipeInfo(pipe):
    return pipe[0]['x'], pipe[0]['y'], pipe[1]['x'], pipe[1]['y']

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1 or player['y'] < 0:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
