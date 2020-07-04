"""
Description:
    Some helper functions are implemented here.
    You can implement your own plotting function if you want to show extra results :).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot(sum_reward, label,ylabel):
    """
    Function to plot the results.
    
    Input:
        avg_reward: Reward averaged from multiple experiments. Size = [exps, timesteps]
        label: label of each line. Size = [exp_name]
    
    """
    assert len(label) == sum_reward.shape[0]

    # define the figure object
    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(111)

    # We show the reward curve
    steps = np.shape(sum_reward)[1]

    for i in range(len(label)):
        ax1.plot(range(steps), sum_reward[i], label=label[i])
    ax1.legend() 
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(ylabel)
    ax1.grid('k', ls='--', alpha=0.3)

    plt.show()

