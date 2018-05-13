#!/usr/bin/env python

import sys
sys.path.append('./FlappyBird')
import pygame, sys, os, random
from config import *
from FlappyBird import FlappyGame

def start(model, testing = False):
	FlappyGame(testing).run(model = model)

# just play flappy bird with a pretrained model
if __name__ == "__main__":
	if len(sys.argv) == 1:
		start(model = MODEL_PATH, testing = True)
	elif len(sys.argv) == 2:
		start(model = sys.argv[1], testing = True)