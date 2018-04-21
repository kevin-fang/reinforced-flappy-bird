#!/usr/bin/env python
import os, sys
from FlappyBird import FlappyGame
from config import *
from shutil import rmtree

# can be imported for automated reinforcement learning
def generate_games(model = False):
    FlappyGame().run(model)

#if os.path.exists(TRAIN_SCREEN_DIR):
#		rmtree(TRAIN_SCREEN_DIR)
# usage: python generate_flappy_data.py OR python generate_flappy_data.py <model name>
# e.g. python generate_flappy_data.py models/trained_flappy
if __name__ == "__main__":
    if len(sys.argv) == 1:
        FlappyGame().run(model = False)
    elif len(sys.argv) == 2:
        FlappyGame().run(model = sys.argv[1])
