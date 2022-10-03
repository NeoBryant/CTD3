#coding=utf-8
# File      :   utils.py
# Time      :   2022/09/18 21:44:23
# Author    :   Jinghan Peng
# Desciption:   all kinds of functions

import os, sys
import yaml
import shutil
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties  # 导入字体模块

def load_yml(path):
    """load yaml file"""
    with open(path, 'r', encoding='utf-8') as rf:
        config = yaml.load(rf.read(), Loader=yaml.FullLoader)
    return config


def lmap(v, x, y): # -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def normalize_observation(observations, features, features_range, clip=False):
    """ 对观察值进行正则化到 [-1,1]之间
    Args:
        observations (np): 观测值
        features (list): 观测值的类型
        features_range (dict): 需要正则化的观察值的上下界
    Returns:
        np: 正则化后的观察值
    """
    for index, (feature) in enumerate(features): # 遍历每一种观测值
        if feature in features_range:
            f_range = features_range[feature]

            observations[:,index] = lmap(observations[:,index], [f_range[0], f_range[1]], [-1, 1])
            if clip:
                observations[:,index] = np.clip(observations[:,index], -1, 1)
    
    return observations

def make_dir(project_path, model_config=None, env_config=None):
    """create the dir of project to save the models and results"""
    if os.path.exists(project_path):
        print("Project already existed! Do you want to create it? [Y/N]")
        if input() in ('N', 'n'):
            exit(0)
    
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "trajectory"), exist_ok=True)
    
    if os.path.exists(model_config):
        shutil.copy(model_config, project_path)
    if os.path.exists(env_config):
        shutil.copy(env_config, project_path)
        

def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
        fname='/System/Library/Fonts/STHeiti Light.ttc', size=15) # fname系统字体路径，此处是mac的
    except:
        font = None
    return font

def plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(plot_cfg.env_name,
              plot_cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if plot_cfg.save:
        plt.savefig(os.path.join(plot_cfg.result_path,f"{tag}_rewards_curve_cn"))
    # plt.show()


def plot_rewards(rewards, plot_cfg, step_list, tag='train', xlabel="Step", ylabel="Reward"):
    """plot the reward of each epoch"""
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"
    
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(step_list, rewards, label='epoch_rewards')
    # plt.plot(ma_rewards, label='ma_rewards')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"],"{}_rewards_curve".format(tag)))
    # plt.show()
    plt.close()

def plot_outoflanes(ep_outoflanes, plot_cfg, step_list, tag='train', xlabel="Step", ylabel="Number of Out of Lane"):
    """plot the count of out of lane for each epoch"""
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"
    
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(step_list, ep_outoflanes, label='epoch_outoflanes')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"],"{}_outoflanes_curve".format(tag)))
    # plt.show()
    plt.close()
    
def plot_relative_distance_of_center_line(ep_outoflanes, plot_cfg, step_list, tag='train', xlabel="Step", ylabel="Distance"):
    """plot the """
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"
    
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("distance curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if step_list:
        plt.plot(step_list, ep_outoflanes, label='distance')
    else:
        plt.plot(ep_outoflanes, label='distance')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"],"{}_distance_curve".format(tag)))
    # plt.show()
    plt.close()
    

def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()


def single_plot(x, y, path, xlabel="", ylabel="", title="", label=None):
    """simple plot for any data"""
    sns.set()
    plt.figure() 
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if x:
        plt.plot(x, y, label=label)
    else:
        plt.plot(y, label=label)

    plt.legend()
    plt.savefig(path)
    # plt.show()
    plt.close()
    

def save_results(rewards, costs, episode_steps, tag='train', path='./results'):
    ''' 保存奖励，平滑奖励，驶出车道次数
    '''
    np.save(os.path.join(path,'{}_rewards.npy'.format(tag)), rewards)
    # np.save(os.path.join(path,'{}_ma_rewards.npy'.format(tag)), ma_rewards)
    np.save(os.path.join(path,'{}_costs.npy'.format(tag)), costs)
    
    np.save(os.path.join(path,'{}_episode_steps.npy'.format(tag)), episode_steps)

    
    print('结果保存完毕！')
