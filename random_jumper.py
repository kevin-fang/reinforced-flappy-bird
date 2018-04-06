import random

'''
- Want frames alive to be the reward
- Feed in Flappy Bird image + last jumps to convolutional (?) neural net, have it predict whether to jump or stay
'''

# image takes input of the screenshot, last_jump is the number of frames since the last jump
def get_jump(img, last_jump):
	return random.choice([True, False, False, False, False, False, False, False, False, False, False, False, False])