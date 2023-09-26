# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:09:16 2023

@author: tobyk
"""


"""import modules"""
import random #random number generator
import numpy as np #exp, append, array
import matplotlib.pyplot as plt #scatter plot
import time #check speed of program

plt.close('all')

"""only variables (N) and (iterations) should be changed"""
N = 10 # test N=10
iterations = 1000000 # test for 1,000,000 iterations

k = 1
m = 1
#lowest_temp = 1/N + 0.04
lowest_temp=0.5
beta = 1/lowest_temp
epsilon = beta/N
max_number_points_updated = N-1

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
       
        """Consecutive points"""
        n_updated = random.randint(1,max_number_points_updated) #number of consecutive points to update
       
        """Choose which x value to update & update"""
        point_selected = random.randint(0,N-1)
        updateVal = random.uniform(-1,1)
        #for i in range(n_updated):
       #     x_values
        x_values[-1,point_selected:point_selected+n_updated] += updateVal
        if point_selected+n_updated > N:
            x_values[-1,0:(point_selected+n_updated)%N] += updateVal
       
        """Change in action"""
        changeInAction = m/(2*epsilon)*((x_values[-1,point_selected%N]-x_values[-1,(point_selected-1)%N])**2+(x_values[-1,(point_selected+n_updated)%N]-x_values[-1,(point_selected+n_updated-1)%N])**2-(x_values[-2,point_selected%N]-x_values[-2,(point_selected-1)%N])**2-(x_values[-2,(point_selected+n_updated)%N]-x_values[-2,(point_selected+n_updated-1)%N])**2)
        for j in range(n_updated):
            changeInAction += 0.5*k*epsilon*(x_values[-1,(point_selected+j)%N]**2-x_values[-2,(point_selected+j)%N]**2)
       
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
    mean_split = np.delete(mean_split,0,axis=0)
    mean_split = np.hstack(mean_split)
    return mean_split


positions2 = updateX(beta, N, m, k)


"""plot figure"""
fig = plt.figure(figsize=(13,9))
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set_xlabel('position x',fontsize =15)
ax1.set_ylabel('probability density',fontsize=15)
ax1.set_xlim(-3,3)
ax1.tick_params(axis='both', labelsize=15)

ax1.hist(positions2, bins=100, density=True)

x_axis = np.arange(-3,3,0.01)

def func(x):
    return (np.sqrt(1/np.sqrt(np.pi))*np.exp(-x**2/2))**2

ax1.plot(x_axis, func(x_axis))
