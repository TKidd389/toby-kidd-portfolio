# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:30:20 2022

@author: tobyk
"""


"""import modules"""
import numpy as np # exp, pi, sqrt
import matplotlib.pyplot as plt # plot wave
from scipy.sparse import diags # matrix of second order position derivative
from scipy.integrate import odeint
import time # time code runs for

"""close pre-existing plots"""
plt.close('all')

"""inital values"""
x0 = -10 # centre of wave packet
sigma = 2 # width of wave packet

N_discretised = 501 # number of x values computed
width = 25

V0 = 1
E = 2
hBar = 1
m = 1
k0 = np.sqrt(2*m*E)/hBar

summation = np.array([])
std_errs = np.array([])
for i in range(1,100):
    def psi_t0(x):
        return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)
    
    """discretised initial wave packet"""
    x_values = np.linspace(-width,width,N_discretised)
    wavePacket = psi_t0(x_values)
    probabilityDistribution = abs(wavePacket)**2
    
    """potential as a matrix"""
    def potential(x):
        """
        a = np.copy(x)
        a[a<0] = 0
        a[a>1]=0
        a = np.where(a != 0, 0, a)
        return a
        """
        return np.zeros(N_discretised)
    V = np.diag(np.full(N_discretised,potential(x_values)))
    
    """matrix of second order position derivative"""
    M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()
    
    """solving ODE"""
    # time points
    maxTime = 22/k0
    t = np.linspace(0,maxTime,i*10)
    
    # concatanate real and imaginary components
    wavefunction_concatanated = np.concatenate((wavePacket.real,wavePacket.imag), axis=None)
    
    G = (-hBar**2)/(2*m)*M + V
    
    # block matrix
    B = np.block([[np.zeros((N_discretised,N_discretised)),G],[-G,np.zeros((N_discretised,N_discretised))]])
    
    # function that returns dpsi/dt
    def rhs(psi,t):
        return 1/hBar*(np.matmul(B,psi))
    
    start_time = time.time()
    # solve ODE
    all_wavefunction_conc = odeint(rhs, wavefunction_concatanated, t)
    final_wavefunction_conc = all_wavefunction_conc[-1]
    final_wavefunction_conc_split = np.split(final_wavefunction_conc,2)
    final_wavefunction = final_wavefunction_conc_split[0]+1j*final_wavefunction_conc_split[1]
    
    """discretised final wave packet"""
    probabilityDistribution_final = abs(final_wavefunction)**2
    
    end_time = time.time()
 
    summation = np.append(summation, np.sum(probabilityDistribution_final)*(2*width/N_discretised))
 
"""create figure"""
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax1.set_xlabel('Number of Time Steps')
ax1.set_ylabel('Norm of Wavefunction')

time = np.arange(10,1000,10)

ax1.plot(time[1:19], summation[1:19], 'b*')

ax2 = fig.add_subplot(212)
ax2.set_xlabel('Number of Time Steps')
ax2.set_ylabel('Norm of Wavefunction')

time = np.arange(10,1000,10)

ax2.plot(time[1:len(time)], summation[1:len(summation)], 'b*')