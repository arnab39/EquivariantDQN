# EquivariantDQN
Deep equivariant Q-learning for symmetric games

## Required libraries
- python 3.6.9
- pytorch/torch
- pygame
- PIL
- PLE 

### To install PLE, follow these steps
```
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git

cd PyGame-Learning-Environment

pip install -e .
```

## How to run
Run using the following script
```python main.py```


## Running code on custom games, networks 

- The current settings of the code works for the Snake game offered by PLE. Should you wish to use another game offered by PLE, simply change the `snake_env.py` code to replace the `SnakeEnv` class with the class for your preffered game. 
- Currently a simply CNN (with 3 layers) is used for learning the game features. In order to use your own network, change `network.py`. This code also implements the DQN algorithm. Any other algorithm can also be used by modifying it. 
- An arbitrary loss function may be used by modifying the `calculate_td_loss` function in `loss.py`.
- All initialization values like `buffer size`, `batch size`, `gamma`, 	`epsilon`, `number of steps`, etc can be changed and experimented with in `main.py`.  


