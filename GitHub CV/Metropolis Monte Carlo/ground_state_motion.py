# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:28:18 2022

@author: tobyk
"""


"""import modules"""
import random #random number generator
import numpy as np #exp, append, array
import matplotlib.pyplot as plt #scatter plot
import time #check speed of program
import math #rounding

plt.close('all')

"""values
k = 1
beta = 1
m = 1
N = 1
epsilon = beta/N
"""
iterations = 100000

"""functions"""
def updateX(beta,N,m,k):
    epsilon = beta/N
    action = 0
    """Means"""
    means = np.zeros([iterations])
    """Pick arbitrary value for x"""
    x_values = np.array([np.zeros([N])])
    for i in range(iterations):
        """duplicate array"""
        x_values = np.append(x_values,x_values,axis=0)
        
        """Choose which x value to update & update"""
        point_selected = random.randint(0,N-1)
        x_values[-1,point_selected] += random.uniform(-1,1)
        
        """Change in action"""
        changeInAction = m/(2*epsilon)*((x_values[-1,point_selected%N]-x_values[-1,(point_selected-1)%N])**2+(x_values[-1,(point_selected+1)%N]-x_values[-1,point_selected%N])**2-(x_values[-2,point_selected%N]-x_values[-2,(point_selected-1)%N])**2-(x_values[-2,(point_selected+1)%N]-x_values[-2,point_selected%N])**2) + 0.5*k*epsilon*(x_values[-1,point_selected%N]**2-x_values[-2,point_selected%N]**2)
        
        """Test update"""
        if changeInAction <= 0:
            action += changeInAction
            x_values = np.delete(x_values,-2,0)
        else:
            prop = np.exp(-changeInAction)
            if random.random() <= prop:
                action += changeInAction
                x_values = np.delete(x_values,-2,0)
            else:
                x_values = np.delete(x_values,-1,0) # changes updated point back to original value
        means[i] += np.mean(x_values)
    mean_split = np.split(means,10)
    mean_split = np.delete(mean_split,0)
    return means

"""averages function"""
def meanError(means):
    mean_split = np.split(means,10)
    mean_split = np.delete(mean_split,0)
    mean_split = np.split(mean_split,9)
    mean_split_means = np.zeros(9)
    for i in range(0,9):
        mean_split_means[i] += np.mean(mean_split[i])
    overall_mean = np.mean(mean_split_means)
    return overall_mean


positions = np.round(updateX(1/0.05,10,1,1),2)
position, count = np.unique(positions, return_counts=True)
probability = count/iterations


"""plot figure"""
fig = plt.figure(figsize=(9,9))

ax1 = fig.add_subplot(111)
ax1.set_xlabel('<x>')
ax1.set_ylabel('probability')
ax1.set_title('Ground State Motion')
ax1.grid()
ax1.plot(position,probability,marker='x',linestyle='none')