import test_flappy_results
import parse_training_data
from shutil import rmtree
from config import *
import os

def run():
	if os.path.exists(TRAIN_SCREEN_DIR):
		rmtree(TRAIN_SCREEN_DIR)
	test_flappy_results.start(MODEL_PATH)
	parse_training_data.parse_data()

if __name__ == "__main__":
	run()