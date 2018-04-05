# Playing Flappy Bird with Machine Learning

### Requirements:

OpenCV (`conda install -c conda-forge opencv`), NumPy (`pip install numpy`), PyGame (`pip install pygame`), TFLearn (`pip install tflearn`), TensorFlow (`pip install tensorflow-gpu`/`pip install tensorflow` not recommended)

### Info

Flappy bird implementation written in Python with PyGame from https://github.com/Max00355/FlappyBird, significantly modified to work with code.

`nn_model.py` holds the neural network model  
`random_jumper.py` contains the initial flappy bird player (randomly jumps with a bias)  
`train_nn.py` pulls out training and testing data from the Flappy Bird game and trains the network  
`config.py` contains some configuration information (headless mode is really slow, for some reason)