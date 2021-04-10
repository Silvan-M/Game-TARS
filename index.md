# Game-TARS
The goal of this project is to train a deep learning AI to play games using Deep-Q-Learning.

# Installation
Make sure you have following libraries installed:
- Numpy
- TensorFlow
- Pygame

Then simply clone the project.

# Training
There are already pretrained models which we created for you. If you don't want to go through the hussle of training your own, skip this part.
To train a game regurarly go into the main folder and launch `train_dqn.py`. Then you can type in one of our games ("tictactoe", "snake" or "spaceinvader") followed by the amount of games you want to train.
You can also launch `train_dqn_vs_dqn.py` to train AI vs AI (only available for TicTacToe).

# Playing
To play any of our games go into the main folder and launch `play_dqn_pygame.py`. A user interface will pop up and you can choose one of three games. 
You are able to choose to let the AI play or you can try the game yourself. When choosing AI you will be asked for a model. Choose a model you trained earlier or use our pretrained model for the best experience.

# Showcase Video
If you want to check out how the AI performs without downloading our project, watch our demo (click the gif to watch the full video):

[![Game TARS - Showcase](https://j.gifs.com/P7vOjl.gif)](https://youtu.be/2EOwnuxdsIs)
