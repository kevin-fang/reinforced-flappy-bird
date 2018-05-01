# Playing Flappy Bird with Reinforcement Learning

## Running

After installing the requirements (NumPy, PyGame, and TensorFlow), run `python train_nn.py -n` to generate a new model. It will play 20 games of Flappy Bird, train, and do that again, retraining each round. If you have a saved model, run `python train_nn.py -l` and it will retrain on the model already available in the folder `./models`. 

Once the model has been trained, it can be tested with `python test_flappy_results.py <model path>` and it will play the game with the existing model. 

Some pretrained models exist in `./saved_models`

### Requirements:

Numpy (`pip install numpy`), PyGame (`pip install pygame`), TensorFlow (`pip install tensorflow-gpu`/`pip install tensorflow`), Python 3.

### Info

Flappy bird implementation written in Python with PyGame from https://github.com/Max00355/FlappyBird, significantly modified to work with machine learning code.

`python train_nn.py -n` will generate a new model with random initial weights and keep training in iterations of 20 games. It will save a model in a new folder `models/`.

`python test_flappy_results.py <model_name>` will play Flappy Bird games with an initially trained model.

`neural_jumper.py` is a wrapper for the neural network to choose a jump.

`tf_graph.py` contains the neural network architecture.

`config.py` contains some configuration variables, such as whether to save images and directories.

`global_vars.py` contains some global variables - it should not be modified.

