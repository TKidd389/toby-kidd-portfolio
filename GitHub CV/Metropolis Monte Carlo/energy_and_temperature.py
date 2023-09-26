# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:56:23 2022

@author: tobyk
"""


"""import modules"""
import random #random number generator
import numpy as np #exp, append, array
import matplotlib.pyplot as plt #scatter plot
import time #check speed of program

plt.close('all')

"""define variables"""
"""only variables (N) and (iterations) should be changed"""
N = 3 # test values should be set to 2 or 3, due to time constraints
iterations = 100000 # values should be set to powers of 10, with a test value of 100000

k = 1
beta = 1
m = 1
epsilon = beta/N

"""Function updates x using an iterative procedure using random numbers and probability"""
def updateX(beta,N,m,k):
    epsilon = beta/N
    action = 0

    means = np.zeros([iterations])
    
    """Pick arbitrary value for x"""
    x_values = np.array([np.zeros([N])])
    
    for i in range(iterations):
        """Duplicate array"""
        x_values = np.append(x_values,x_values,axis=0)
       
        """Choose which x value to update & update"""
        point_selected = random.randint(0,N-1)
        x_values[-1,point_selected] += random.uniform(-1,1)
       
        """Change in action"""
        changeInAction = m/(2*epsilon)*((x_values[-1,point_selected%N]-x_values[-1,(point_selected-1)%N])**2+(x_values[-1,(point_selected+1)%N]-x_values[-1,point_selected%N])**2-(x_values[-2,point_selected%N]-x_values[-2,(point_selected-1)%N])**2-(x_values[-2,(point_selected+1)%N]-x_values[-2,point_selected%N])**2) + 0.5*k*epsilon*(x_values[-1,point_selected%N]**2-x_values[-2,point_selected%N]**2)
       
        """Test update: choose whether to accept or reject the updated x value"""
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
        means[i] += np.mean(x_values**2)
    return means

"""averages function: used to return the statistics of the final result/array of accepted x values"""
def meanError(means):
    mean_split = np.split(means,10)
    mean_split = np.delete(mean_split,0)
    mean_split = np.split(mean_split,9)
    mean_split_means = np.zeros(9)
    for i in range(0,9):
        mean_split_means[i] += np.mean(mean_split[i])
    std_err = np.std(mean_split_means)/(np.sqrt(len(mean_split_means))-1)
    overall_mean = np.mean(mean_split_means)
    return overall_mean, std_err

"""plot figure"""
fig = plt.figure(figsize=(9,9))

"""plot <x^2> against Temperature"""
means = np.array([])
std_errs = np.array([])
temperature_values = np.array([i for i in range(1,21)])/20
start_time = time.time()
for i in range(0,20):
    meanErrors = meanError(updateX(1/temperature_values[i],N,m,k))
    means = np.append(means,meanErrors[0])
    std_errs = np.append(std_errs,meanErrors[1])
end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time)) # used to check the efficiency of the program

ax1 = fig.add_subplot(111)
ax1.grid() # adds a grid to the plot
ax1.errorbar(temperature_values,means, yerr=[std_errs,std_errs], capsize=5, fmt="o")
ax1.set_xlabel('Temperature')
ax1.set_ylabel('<x^2>')
ax1.title.set_text('<x^2> against Temperature')

"""plot expected curve"""
expected = (np.exp(1/temperature_values)+1)/(2*(np.exp(1/temperature_values)-1))
ax1.plot(temperature_values,expected)

"""calculate the percentage of simulated data points within error bars of expected values"""
inErrorBars = 0
for i in range(len(means)):
    if means[i]-std_errs[i]<=expected[i] and means[i]+std_errs[i]>=expected[i]:
        inErrorBars += 1
inErrorsPercentage = inErrorBars/len(means)*100
#ax1.text(0,1,str(inErrorsPercentage)+'% within error bars',fontsize=15,backgroundcolor='w')

avgError = np.mean(std_errs) # used as a check