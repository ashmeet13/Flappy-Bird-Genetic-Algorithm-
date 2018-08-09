# Flappy-Bird-Genetic-Algorithm
This is a simple python code that uses neuro-evolution to learn how to play Flappy Bird

There are two python files included:

Game.py:

This is a simple game that can be played raw using the "space" key to jump.
It will use the "index2.png" file. Download the image and keep it in the same folder as Game.py

FlappyBird.py:

This is the genetic algorithm implementation for the game.
This python file uses "index.png" image. Download this image if you are running the FlappyBird.py

Libraries used:

    pygame
    math
    random
    os
    numpy
    
Genetic Algorithm:

Basically, the game is executed once with a certain number of birds (The Population). After this generation of the population finishes playing the game(i.e. every bird has lost) the code will choose the best bird(Choose the parameters upon which the it was deciding whether to jump or not jump). The best bird will be choosen using a fitness equation which in this case is:
    
    Distance Travelled by the Bird - (Distance between the center of the bird and the center of the gap between the pipes.

After the best bird parameters are selected, 10% of the birds in the next generation will be same as the best bird of previous generation. The remaining 90% of the birds would be created using mutation.

The process repeats until and unless you get a near perfect bird, if not perfect.
