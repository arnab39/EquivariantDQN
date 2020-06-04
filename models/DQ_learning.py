import math
import torch
import collections
import numpy as np
from memory.replay_memory import replay_buffer
from memory.prioritized_replay_memory import prioritized_replay_buffer
from tqdm import tqdm
import os

__all__ = ["DQN"]

class DQN():
    def __init__(self, device, network, target_network, optimizer, environment, gamma, batch_size, replay_memory_size, epsilon_decay,
                            checkpoint_dir, Double_DQN, epsilon_end, priority_replay=False, alpha=0.6, beta_start=0.4, beta_frames = 100000, replay_initial = 1000):
        self.device = device
        self.network = network
        self.target_network = target_network
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
        self.Double_DQN = Double_DQN
        if self.Double_DQN:
            print("You are using Double DQN")
        self.epsilon_final = epsilon_end
        self.priority_replay = priority_replay
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.replay_initial = replay_initial
        self.beta = beta_start

    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1.0
        return self.epsilon_final + (epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def update_target(self):
        self.network.eval()
        self.target_network.load_state_dict(self.network.state_dict())
        self.network.train()

    def update_network(self):
        if self.priority_replay:
            cur_states, actions, next_states, rewards, masks, indices, weights = self.buffer.sample(self.batch_size, self.beta)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        else:
            cur_states, actions, next_states, rewards, masks = self.buffer.sample(self.batch_size)
        cur_states = torch.stack(cur_states).type(dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).type(dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.int8).to(self.device)
        masks = torch.tensor(masks, dtype=torch.int8).to(self.device)

        Q_values = self.network(cur_states)
        next_Q_values = self.target_network(next_states)

        Q_values = torch.gather(Q_values, 1, actions.unsqueeze(1)).squeeze(1)
        if self.Double_DQN:
            next_Q_values_select = self.network(next_states)
            next_Q_values = next_Q_values.gather(1, torch.max(next_Q_values_select, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            next_Q_values = next_Q_values.max(1)[0]

        delta = rewards + self.gamma * next_Q_values.detach() * masks - Q_values

        if self.priority_replay:
            loss = delta.pow(2) * weights
            loss = loss.mean()
            prios = torch.abs(delta) + 1e-5
            self.buffer.update_priorities(indices, prios.data.cpu().numpy())
        else:
            loss = delta.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_model(self,total_episodes, frame_update, writer):
        episode_count = 1
        episode_length = 0
        episode_reward = 0
        self.network.train()
        cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
        for frame in tqdm(range(3000000),desc="Training"):
            epsilon = self.epsilon_by_frame(frame)
            action = self.network.get_action(cur_state, epsilon)
            next_state, reward, done, _ = self.environment.take_action(action)
            cur_state = cur_state.type(dtype=torch.uint8).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.uint8).to(self.device)
            episode_length = episode_length + 1
            episode_reward = episode_reward + reward
            mask = 0 if done else 1
            self.buffer.push(self.transition(cur_state, action, next_state, reward, mask))
            if self.priority_replay == True and len(self.buffer) >= self.replay_initial:
                self.beta = self.beta_by_frame(frame)
                loss = self.update_network()
                writer.add_scalar('Loss', loss, frame)
            if self.priority_replay == False and len(self.buffer) >= self.batch_size:
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
            if frame % frame_update == 0:
                self.update_target()
            if episode_count > 2000 and episode_count % 250 == 249:
                self.network.eval()
                torch.save({
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, self.checkpoint_path)
                self.network.train()
            if episode_count == total_episodes:
                break

    def eval_model(self,total_episodes):
        episode_count = 1
        cumulative_reward = 0
        checkpoint = torch.load(self.checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
        for frame in tqdm(range(1000000), desc="Evaluating"):
            action = self.network.get_action(cur_state, 0.01)
            next_state, reward, done, _ = self.environment.take_action(action)
            cumulative_reward = cumulative_reward + reward
            cur_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            if done:
                episode_count = episode_count + 1
                cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
            if episode_count == total_episodes:
                break
        return cumulative_reward/total_episodes
