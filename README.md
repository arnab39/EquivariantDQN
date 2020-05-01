# EquivariantDQN
![Snake Game](assets/snake.gif)

This is the official repository of our paper "Steerable CNNs for Deep Q-Learning in symmetric environments". Note that the entire code is written in python 3.7 to make it compatible with e2cnn library. We use the Snake environment throughout the project. 

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
Although we give user the choice to play with the model by allowing them to select different arguments we would recommend them to keep most of them to default which are found to be the best hyperparameter. Also user should note that in our implementation we decay `epsilon` to 0.01 from 1 and the decay rate can be controlled by hyperparameter `epsilon_decay`.
1) To run a Vanilla CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type regular  ```<br/>
Alternatively, user can go to main.py and change the default arguments. We give a default of 4000 for totat_episodes which user can change as per their requirement by adding ```--total_episodes new_number```. 
2) To run a D_4 Equivariant CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type D4_Equivariant ```<br/>
3) To run a prioritized replay based Vanilla CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type regular --priority_replay True ```<br/>
4) To run a prioritized replay based D_4 Equivariant CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type D4_Equivariant --priority_replay True ```<br/>

## Recreating the results in the paper
<p float="left">
  <img src="https://github.com/arnab39/EquivariantDQN/blob/master/assets/Eq_vs_reg_cnn_normal_replay.png" width="400" />
  <img src="https://github.com/arnab39/EquivariantDQN/blob/master/assets/Eq_vs_reg_cnn_priority_replay.png" width="400" />

Once user run the code they will have the rewards per episode stored in the events file of tensorboard which then can be downloaded as csv files. To recreate the results we recommend running all the four models with a seed of 1,2 and 3 using the default hyperparameters. Then the csv obtained from them should be put in csv_files folder and can be plotted running the plot_files.py file. Infact, we also provide the csv files of all the runs.<br/>
To recreate the results in the table take average reward over last 500 episodes for each of the model from the csv files obtained. We also provide the final checkpoint of our four primary models which is obtained after training for 4000 episodes. User can run the following to evaluate the model.<br/>
```python main.py --seed checkpointseed --total_episodes num_episodestest --checkpoint_dir './checkpoints/dir_name' --network_type network_chosen ```<br/>
It can be tested for any number of episodes and returns the average reward obtained from all those. 
</p>


## Running code on custom games and networks 

- The code has been written in a modular way. The current settings of the code works for the Snake game (we also provide code for Cartpole for debugging the learning algorithm) offered by PLE. If user wishes to use another game, he/she has to add `othergame_env.py` and implement the game class according the member functions of Snake class. 
- Currently Vanilla and Equivariant networks have been implemented in the network folder. In order to use another network, the user needs to add `othernetwork.py`and implement the network class according to the member functions of either of Vanilla or Equivariant network.
- We recommend to keep the buffer implementations and DQ learning class same and while letting users to play with the hyperparameters for them.
- All initialization values like `buffer size`, `batch size`, `gamma`, 	`epsilon_decay`, `total_episodes`, etc can be found and changed in `main.py` or from terminal by using arguments.  


