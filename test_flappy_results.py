#!/usr/bin/env python

import pygame, sys, os, random
from config import *
from FlappyBird import FlappyGame

# just play flappy bird with a pretrained model
if __name__ == "__main__":
    if len(sys.argv) == 1:
        FlappyGame().run(model = MODEL_PATH)
    elif len(sys.argv) == 2:
        FlappyGame().run(model = sys.argv[1])