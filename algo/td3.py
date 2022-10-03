import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


seed = 1024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = [] 
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)'''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) 
        state, action, reward, next_state, done =  zip(*batch) 
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量'''
        return len(self.buffer)

    

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1):
        super(Actor, self).__init__()

        
        
        
        self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
        )

        

    def forward(self, state):
        action = self.fc(state)
        action = torch.tanh(action)
        
        
        return action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        
        
        
        self.fc = nn.Sequential(
                nn.Linear(state_dim+action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        
        
        

        q_value = self.fc(state_action)
        
        return q_value


class TD3():
    def __init__(self, cfg, device, writer=None, max_action=1):
        
        self.model_config = cfg["model"]
        state_dim  = self.model_config["observation_dim"]
        action_dim = self.model_config["action_dim"]
        hidden_dim = self.model_config["hidden_dim"]
        
        self.max_action = max_action
        
        self.batch_size = cfg["dataloader"]["batch_size"]
        self.soft_tau = cfg["trainer"]["soft_tau"] 
        self.gamma = cfg["trainer"]["gamma"]
        
        self.policy_delay = cfg["model"].get("policy_delay", 2)
        self.policy_noise = cfg["model"].get("policy_noise", 0.2)
        self.noise_clip = cfg["model"].get("noise_clip", 0.5)
        
        
        if device == "-1":
            self.device = torch.device("cpu")  
        else:
            self.device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu")  
        
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg["trainer"]["actor_lr"])
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=cfg["trainer"]["critic_lr"])
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=cfg["trainer"]["critic_lr"])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        print(self.actor)
        print(self.critic_1)
        print(self.critic_2)
        
        
        self.memory = ReplayBuffer(cfg["dataloader"]["memory_capacity"])
        
        
        self.writer = writer 
        
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        


    def choose_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration=1):
        """更新网络参数
        Args:
            num_iteration (int, optional): 网络批处理学习次数. Defaults to 1.
        """
        
        if len(self.memory) < self.batch_size: 
            return
        
        
        for i in range(num_iteration):    
            
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)
            
            state = torch.FloatTensor(np.array(state)).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
            
            action = torch.FloatTensor(np.array(action)).to(self.device)
            reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

            
            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            

            
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q  = torch.min(target_Q1, target_Q2)
            target_Q  = reward + ((1 - done) * self.gamma * target_Q).detach()

            
            current_Q1 = self.critic_1(state, action)
            loss_Q1    = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            
            current_Q2 = self.critic_2(state, action)
            loss_Q2    = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            
            self.num_critic_update_iteration += 1
            
            
            if self.num_critic_update_iteration % self.policy_delay == 0:
                
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        ((1- self.soft_tau) * target_param.data) + self.soft_tau * param.data
                    )
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(
                        ((1 - self.soft_tau) * target_param.data) + self.soft_tau * param.data
                    )
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(
                        ((1 - self.soft_tau) * target_param.data) + self.soft_tau * param.data
                    )

                self.num_actor_update_iteration += 1
        
        self.num_training += 1

    def save(self, path):
        """save models"""
        torch.save(self.actor.state_dict(), os.path.join(path,'actor.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path,'actor_target.pth'))
        torch.save(self.critic_1.state_dict(), os.path.join(path,'critic_1.pth'))
        torch.save(self.critic_1_target.state_dict(), os.path.join(path,'critic_1_target.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(path,'critic_2.pth'))
        torch.save(self.critic_2_target.state_dict(), os.path.join(path,'critic_2_target.pth'))
        print("model has been saved...")

    def load(self, path):
        """load models"""
        self.actor.load_state_dict(torch.load(os.path.join(path,'actor.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(path,'actor_target.pth')))
        self.critic_1.load_state_dict(torch.load(os.path.join(path,'critic_1.pth')))
        self.critic_1_target.load_state_dict(torch.load(os.path.join(path,'critic_1_target.pth')))
        self.critic_2.load_state_dict(torch.load(os.path.join(path,'critic_2.pth')))
        self.critic_2_target.load_state_dict(torch.load(os.path.join(path,'critic_2_target.pth')))
        print("model has been loaded...")
        

