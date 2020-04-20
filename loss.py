# This file contains the code to calculate and propogate the TD loss

import torch.autograd as autograd
import torch
import numpy as np

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def compute_td_loss(model, replay_buffer, optimizer, batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) # sample from the replay buffer

    # convert all variables into as suitable for CUDA
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state) # get the q-values of current state
    next_q_values = model(next_state) # get the q-values of next state

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done) # calculate expected q-value
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean() # calculate the loss between actual q-value and expected
    
    # backpropogation of loss
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    
    return loss