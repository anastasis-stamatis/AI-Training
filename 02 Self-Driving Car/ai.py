
# coding: utf-8

# # Artificial Intelligence
# Building the ANN

# In[1]:


# Importing libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# In[2]:


# Creating the Neural Network architecture
class Network(nn.Module):
    
    # init function for a deep QN
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size=input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 75)
        self.fc2 = nn.Linear(75,150)
        self.fc3 = nn.Linear(150,50)
        self.fc4 = nn.Linear(50,nb_action)
    # forward function for backpropagation    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x2))
        q_values = self.fc4(x3)
        return q_values


# In[3]:


# Implementing Experience Replay
    
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    # removing past experiences    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
            
    # getting a random sample for learning
    def sample(self, batch_size):
        # if list = {(1,2,3), (4,5,6)}, then zip(*list) = {(1,4),(2,5),(3,6)}
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# In[4]:


# Implement Deep Q Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(250000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0008) #default lr: 0.001
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
          
    def select_action(self, state):
        #Variable: converting tensor state into torch variable.
        #volatile: we won't be needing the gradient for the output of the first state. saving memory
        #temperature: 0 to 100. the larger, the more certain the AI for the action.
        #temperature; softmax q-values: (1, 2, 3)*3 -> 3 becomes the dominant value so with greater probabilities
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=7
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.data[0]
    
    def learn (self, batch_state, batch_next_state, batch_reward, batch_action):
        # batches consistent to the concantenation we made with respect to the first dimension
        # taking batches from the memory which define our transitions
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        # reinitializing the optimizer at every step
        self.optimizer.zero_grad()
        # back-propagating the error
        td_loss.backward() #retain variables=True: Free some memory.
        # updating the Weights
        self.optimizer.step()
        
    def update (self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # learning through 100 random batches
        if len(self.memory.memory)>100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # updating last action and last state
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len (self.reward_window)>1000:
            del self.reward_window[0]
        return action
        
    def score (self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save (self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    def load (self):
        if os.path.isfile('last_brain.pth'):
            print('loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print ('checkpoint loaded')
        else:
            print ('no checkpoint found')

