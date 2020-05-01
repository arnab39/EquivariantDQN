# EquivariantDQN
This is the official repository of our paper "Steerable CNNs for Deep Q-Learning in symmetric environments". Note that the entire code is written in python 3.7 to make it compatible with e2cnn library.

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
- gym (optional|only for cartpole)

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
Although we give user a choice to play with the model by allowing them to select different arguments while they run we would recommend them to keep most of them to default which are found by hyperparameter tuning.
1) To run a Vanilla CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type regular --train True   ```<br/>
Alternately you can go to main.py and change the default arguments. We give a default of 4000 for totat_episodes which user can change as per their requirement. 
2) To run a D_4 Equivariant CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type D4_Equivariant --train True   ```<br/>
3) To run a prioritized replay based Vanilla CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type regular --priority_replay True --train True   ```<br/>
4) To run a prioritized replay based D_4 Equivariant CNN with seed 1 in GPU 0 <br/>
```python main.py --seed 1 --summary_dir 'runs/dir_name' --checkpoint_dir './checkpoints/dir_name' --gpu_id 0 --network_type D4_Equivariant --priority_replay True --train True   ```<br/>


## Running code on custom games, networks 

- The current settings of the code works for the Snake game offered by PLE. Should you wish to use another game offered by PLE, simply change the `snake_env.py` code to replace the `SnakeEnv` class with the class for your preffered game. 
- Currently a simply CNN (with 3 layers) is used for learning the game features. In order to use your own network, change `network.py`. This code also implements the DQN algorithm. Any other algorithm can also be used by modifying it. 
- An arbitrary loss function may be used by modifying the `calculate_td_loss` function in `loss.py`.
- All initialization values like `buffer size`, `batch size`, `gamma`, 	`epsilon`, `number of steps`, etc can be changed and experimented with in `main.py`.  


