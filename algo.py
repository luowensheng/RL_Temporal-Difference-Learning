import sys
import gym
import itertools
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import copy as cp
from utils import plot

def epsilon_greedy_policy(Q, epsilon, num_of_action):
    """
    Description:
        Epsilon-greedy policy based on a given Q-function and epsilon.
        Don't need to modify this :) 
    """
    def policy_fn(obs):
        A = np.ones(num_of_action, dtype=float) * epsilon / num_of_action
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.

    Inputs:
        env: Environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action.

    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    severe_drops_qlearning=0
    # start training
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state) 
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        total_reward=0
        
        for t in itertools.count():          
       #get next state and reward
            next_state, R, done, info = env.step(action)
            total_reward+=R
            
            if done:
               Q[state][action] += alpha * (R - Q[state][action] ) 
               break
            else:
               next_action=np.argmax(Q[next_state][:])
 
            #Update Q
               Q[state][action]+=alpha*(R+discount_factor*Q[next_state][next_action]-Q[state][action])
               policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)
           
            #update state-action pair
            state=cp(next_state)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            
        #update rewards and episode lengths
        episode_rewards[i_episode]=total_reward
        episode_lengths[i_episode]=(t+1)
        if total_reward<-99:
            severe_drops_qlearning+=1
            
    plot(np.expand_dims(episode_rewards, axis=0), ['q_learning'],"Sum of rewards during episode")        
    print('\nThe number of severe drops for Q-learning: ',severe_drops_qlearning,'\n')        
    print('\nThis is the trajectory for Q-learning:\n')     
    return Q, episode_rewards, episode_lengths

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control.

    Inputs:
        env: environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)
    
    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    severe_drops_sarsa=0
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        total_reward=0
        
        for t in itertools.count():
        
       #get next state and reward
            next_state, R, done, info = env.step(action)
            total_reward+=R
            
            if done:
               Q[state][action] += alpha * (R - Q[state][action] ) 
               break
            else:
             action_probs = policy(next_state)
             next_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
               
            #Update Q
             Q[state][action]+=alpha*(R+discount_factor*Q[next_state][next_action]-Q[state][action])
             
            #update state-action pair
            state=cp(next_state)
            action=cp(next_action) 

        #update rewards and episode lengths      
        episode_rewards[i_episode]=total_reward
        episode_lengths[i_episode]=(t+1) 
        if total_reward<-99:
            severe_drops_sarsa+=1
            
    plot(np.expand_dims(episode_rewards, axis=0), ['sarsa'],"Sum of rewards during episode")        
    print('\nThe number of severe drops for Sarsa: ',severe_drops_sarsa,'\n')                    
    print('\nThis is the trajectory for Sarsa:\n')        
    return Q, episode_rewards, episode_lengths

