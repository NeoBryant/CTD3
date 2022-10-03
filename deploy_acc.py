from json import load
import os, sys
from tqdm import tqdm
import math
import time

import numpy as np
from argparse import ArgumentParser

import gym
import highway_env

import torch
from tensorboardX import SummaryWriter

import utils

from algo import *
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="the path to save project", 
                        default="")
                        
    parser.add_argument("--model_config", type=str, help="path of rl algorithm configuration", 
                        default="conf/algorithm/TD3_acc.yaml")
    
    parser.add_argument("--env_config", type=str, help="path of highway env", 
                        default="conf/env/highway_acc_continuous_acceleration.yaml")
    parser.add_argument("--gpu", type=str, help="[0,1,2,3 | -1] the id of gpu to train/test, -1 means using cpu", 
                        default="-1")
    
    args = parser.parse_args()
    return args


def build_env(env_name, configure):
    """build the highway environment"""
    
    env = gym.make(env_name, render_mode='human')
    env.configure(configure)
    env.reset()
    
    
    return env


def build_agent(config, gpu, writer):
    """choose reinforcement learning algorithm"""
    if config["algo_name"] == "TD3":
        agent = TD3(config, gpu, writer=writer)
    elif config["algo_name"] == "TD3_risk":
        agent = TD3_risk(config, gpu, writer=writer)
    
    return agent

def get_reward(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your reward function here! """
    
    reward = 0
    
    ego_presence, ego_x, ego_vx = state[0]
    lead_presence, lead_relative_x, lead_relative_vx= state[1] 

    
    reward += (0.05 * ego_vx)
    
    
    return reward

def get_cost(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your risk function here! """
    cost = 0
    if info["crashed"]:
        cost = 100
    
    return cost

def eval_policy(agent, eval_env, env_config, algo_config, eval_episodes=10, seed=1024):
    """evalurate the agent during training"""
    average_time_per_step = 0.
    num_steps = 0
    
    with torch.no_grad():
        avg_reward = 0.
        num_crash = 0 
        
        for _ in tqdm(range(eval_episodes)):
            state = eval_env.reset()
            state = utils.normalize_observation(state, 
                                        env_config["config"]["observation"]["features"], 
                                        env_config["config"]["observation"]["features_range"], 
                                        clip=False)
            state = state.flatten()
            state = state[-2:]
            
            done, truncated = False, False
            while not done and not truncated:
                
                beg_time = time.time()
                action = agent.choose_action(np.array(state))
                
                end_time = time.time()
                average_time_per_step += (end_time - beg_time)
                num_steps += 1
                
                if algo_config["model"]["action_dim"] == 1: 
                    real_action = [action[0], algo_config["model"]["action_config"]["steering"]]                
                    real_action = np.array(real_action).clip(eval_env.action_space.low, eval_env.action_space.high)
                
                
                state, reward, done, truncated, info = eval_env.step(real_action)
                if info["crashed"]: 
                    num_crash += 1
                
                
                reward = get_reward(info, state, action, done, truncated)
                avg_reward += reward
                
                state = utils.normalize_observation(state, 
                                        env_config["config"]["observation"]["features"], 
                                        env_config["config"]["observation"]["features_range"], 
                                        clip=False)
                state = state.flatten()
                state = state[-2:]

                
        avg_reward /= eval_episodes
    
    print(f"Average decision time per step: {average_time_per_step/num_steps} s")
    print(f"Average return: {avg_reward}")
    print(f"Num Crash: {num_crash}")
    
    return avg_reward, num_crash


def main():
    config = parse_args()
    

    
    env_config = utils.load_yml(config.env_config)
    env = build_env(env_config["env_name"], env_config["config"])
    
    
    
    algo_config = utils.load_yml(config.model_config)
    agent = build_agent(algo_config, config.gpu, None)
    agent.load(config.checkpoint)
    
    
    eval_policy(agent, env, env_config, algo_config, eval_episodes=1000, seed=1024)
    
    
    
    env.close()
    


if __name__ == '__main__':
    main()
