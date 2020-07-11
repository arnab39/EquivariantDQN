# EquivariantDQN
<img src="/assets/snake.gif" width="250" height="250"/> <img src="/assets/pacman.gif" width="250" height="250"/>

This is the official repository of our paper ["Group Equivariant Deep Reinforcement Learning"](https://arxiv.org/abs/2007.03437). Note that the entire code is written in python 3.7. We used the Snake and Pacman environment throughout the project. 

## Required libraries
- pytorch>=1.4
- pygame
- PIL
- PLE 
- e2cnn
- tqdm
- tensorboard
- numpy
- scipy
- gym (optional | only for cartpole)

### To install PLE, follow these steps
```
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git

cd PyGame-Learning-Environment

pip install -e .
```
### To install e2cnn, follow these steps
```
git clone https://github.com/QUVA-Lab/e2cnn

cd e2cnn

pip install .
```

## How to run
Although you can select different arguments we would recommend to use the default setting which are found by manually tuning hyperparamters. Values those hyperparameter arguments which are different for Snake and Pacman:

- `replay_memory_size` (Snake: 100000| Pacman:300000)
- `frame_update_freq` (Snake: 1000| Pacman:10000)

Other default arguments can be dound in the `main.py` file. As we kept all the setting same for Vanilla and Equivariant network, using other hyperparameters will give similar improvements. Also user should note that in our implementation we decay `epsilon` to 0.01 from 1 and the decay rate can be controlled by hyperparameter `epsilon_decay` which we typically keep to 40,000. Increasing decay this will lead to more exploration at the beginning and possibly better results for both the models. 
 
## Reproducing the results in the paper
<p float="left">
  <img src="https://github.com/arnab39/EquivariantDQN/blob/master/assets/vanilla_vs_eq_snake.png" width="400" />
  <img src="https://github.com/arnab39/EquivariantDQN/blob/master/assets/vanilla_vs_eq_pacman.png" width="400" />
To reproduce these results run experiments from seed 1 to 10 and extract the rewards from tensorboard files. Then use the plotting function to plot them. You can get the plots for 1 seed in tensorboard itself.
</p>


## Running code on custom games and networks 

- The code has been written in a modular way. The current settings of the code works for the Snake game (we also provide code for Cartpole for debugging the learning algorithm) offered by PLE. If user wishes to use another game or any other path planning environment, he/she has to add `othergame_env.py` and implement the game class according the member functions of Snake class. 
- Currently Vanilla and Equivariant networks have been implemented in the network folder. In order to use another network, the user needs to add `othernetwork.py`and implement the network class according to the member functions of either of Vanilla or Equivariant network.
- We recommend to keep the buffer implementations and DQ learning class same and while letting users to play with the hyperparameters for them. 

## Computational Resources
We trained the model in Titan Xp and all the models take 5-6 hours to run for 4000 episodes. 

## Contacts
For more details regarding the code drop an email at: arnabm@cim.mcgill.ca
If you use this code for your research, please consider citing the original paper:

- Arnab Kumar Mondal, Pratheeksha Nair, Kaleem Siddiqi. [Group Equivariant Deep Reinforcement Learning](https://arxiv.org/pdf/2007.03437.pdf), Presented in ICML 2020 Worshop on Inductive Biases, Invariances and Generalization in RL.

