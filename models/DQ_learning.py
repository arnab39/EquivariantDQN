import math
import torch
import collections
import sys
from memory.replay_memory import replay_buffer
from memory.prioritized_replay_memory import prioritized_replay_buffer
from tqdm import tqdm
import os
import numpy as np

__all__ = ["DQN_train"]

class DQN_train():
    def __init__(self, device, network, optimizer, environment, gamma,
              batch_size, replay_memory_size, epsilon_decay, checkpoint_dir, priority_replay=False, alpha=0.6, beta=0.4):
        self.device = device
        self.network = network
        self.optimizer = optimizer
        self.environment = environment
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.buffer = prioritized_replay_buffer(replay_memory_size,alpha) if priority_replay else replay_buffer(replay_memory_size)
        self.epsilon_decay = epsilon_decay
        self.transition = collections.namedtuple('transition', ['cur_state', 'action', 'next_state', 'reward', 'mask'])
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = checkpoint_dir + '/saved_model'
        self.priority_replay = priority_replay
        self.alpha = alpha
        self.beta = beta

    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1.0
        epsilon_final = 0.01
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

    def update_network(self):
        if self.priority_replay:
            cur_states, actions, next_states, rewards, masks, indices, weights = self.buffer.sample(self.batch_size,
                                                                                                    self.beta)

            cur_states = torch.tensor(cur_states, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        else:
            cur_states, actions, next_states, rewards, masks = self.buffer.sample(self.batch_size)
            cur_states = torch.stack(cur_states)
            next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.int8).to(self.device)
        masks = torch.tensor(masks, dtype=torch.int8).to(self.device)

        Q_values = self.network(cur_states)
        next_Q_values = self.network(next_states)

        Q_values = torch.gather(Q_values, 1, actions.unsqueeze(1)).squeeze(1)
        next_Q_values = next_Q_values.max(1)[0]

        delta = rewards + self.gamma * next_Q_values.detach() * masks - Q_values
        if self.priority_replay:
            loss = delta.pow(2) * weights
            prios = loss + 1e-5
            loss = loss.mean()
            self.buffer.update_priorities(indices, prios.data.cpu().numpy())
        else:
            loss = delta.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_model(self,total_frames, writer):
        episode_count = 1
        episode_length = 0
        episode_reward = 0
        cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
        for frame in tqdm(range(total_frames),desc="Training"):
            epsilon = self.epsilon_by_frame(frame)

            self.network.eval()
            action = self.network.get_action(cur_state, epsilon)
            self.network.train()

            next_state, reward, done, _ = self.environment.take_action(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

            episode_length = episode_length + 1
            episode_reward = episode_reward + reward
            mask = 0 if done else 1
            if self.priority_replay:
                self.buffer.push(cur_state, action, next_state, reward, mask)
            else:
                self.buffer.push(self.transition(cur_state, action, next_state, reward, mask))

            if len(self.buffer) >= self.batch_size:
                loss = self.update_network()
                writer.add_scalar('Loss', loss, frame)
            cur_state = next_state
            if done:
                writer.add_scalar('Episode length', episode_length, episode_count)
                writer.add_scalar('Reward per episode', episode_reward, episode_count)
                episode_length = 0
                episode_reward = 0
                episode_count = episode_count + 1
                cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
            if frame > 800000 and frame % 200000 == 199999:
                print("Saving model ...")
                torch.save({
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, self.checkpoint_path)



