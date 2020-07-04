import gym
import pdb
import numpy as np
import argparse
import plotting
import itertools

# import ours 
from env import CliffWalkingEnv
from algo import q_learning, sarsa
from utils import plot

# define algorithm map
ALGO_MAP = {'q_learning': q_learning,
            'sarsa': sarsa}

All=True

def render_trajectory(env, Q):
    """
    Description:
        This is the function for you to render the trajectory in Cliff-Walking Environment.
        You should find different trajectory optimized by SARSA and Q-learning!
    """
    state = env.reset()
    for t in itertools.count():
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
        state = next_state

if __name__ == '__main__':
    
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", "--algo", default='q_learning', 
                        choices=ALGO_MAP.keys(),
                        help="algorithm to use")
    parser.add_argument("-episode", "--episode", default=500,
                        help="Training episode")
    parser.add_argument("-render", "--render", action='store_true',
                        help="visualize the result of an algorithm")
    parser.add_argument("-runAll", "--runAll", action='store_true', 
                        help="run all algorithms")
    args = parser.parse_args()

    # initial environment object
    env = CliffWalkingEnv()

    # start training

    if All:
        
        # run all the algorithms
        label = ['q_learning', 'sarsa']
        _gather_r = np.zeros([len(label), args.episode])
        _gather_l = np.zeros([len(label), args.episode])
        for alg in label:
            idx = label.index(alg)
            Q, epi_reward, epi_length = ALGO_MAP[alg](env, args.episode)
            render_trajectory(env, Q)
            _gather_r[idx] = epi_reward
            _gather_l[idx] = epi_length
        plot(_gather_r, ['Q-learning', 'SARSA'],"Sum of rewards during episode")
        plot(_gather_l, ['Q-learning', 'SARSA'],"Time steps taken per episode")
   
