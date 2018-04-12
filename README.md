# Playing Flappy Bird with Machine Learning

### Requirements:

OpenCV (`conda install -c conda-forge opencv`), NumPy (`pip install numpy`), PyGame (`pip install pygame`), TFLearn (`pip install tflearn`), TensorFlow (`pip install tensorflow-gpu`/`pip install tensorflow` not recommended)

### Info

Flappy bird implementation written in Python with PyGame from https://github.com/Max00355/FlappyBird, significantly modified to work with code.

Flappy bird reinforcement learning agent.

- for each frame, output X_t (image), y_t (move made), r_t (reward at that time step - 0.01 for alive, 1 for through pipe, -1 for death)
- processing step, calculate R_t = sum of gamma from i = 1 to t + i = n, where n is the number of frames per game of gamma^t+i * r_t+i
//- p_i is probability that you did what you did (logit)
- R_t is adjusted reward for action number t is equal to look at very next frame, multiple that by 0.99 (gamma). Look at what happened 2 frames later, multiply by .99^2, 100 frames later, multiply by .99 ^ 100.
- Once calculated R_t, do a round of training (couple of iterations of gradient descent), where loss is -sum R_t log p(Y_i | X_i)
- keep track of end of game

X, Y, r are placeholders. Calculate loss whih is negative sum over all T's of Rt log p(yt given xt), but p yt given xt is just the logit or 1 - the logit.
- Step 1: record X, y, r, Rt. Write to a file. Calculate Rt too after game is finished for each frame of the game
- Once calculated, do a round of training which involves updating tensorflow code to take in X, y, r, R as placehodlers. 
- Calculate the loss and create train step operation based on loss. 
- RMS prop
- Rerun the game.