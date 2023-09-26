# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:44:44 2022

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

x_values = np.linspace(-width,width,N_discretised)

E = 2
hBar = 1
m = 1
k0 = np.sqrt(2*m*E)/hBar
def psi_t0(x,k0):
    return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)

V0 = 24.5
"""potential as a matrix"""
def potential(x,a,V0):
    pot = np.append(np.zeros(int((N_discretised-1)/2)),np.ones(a)*V0)
    pot = np.append(pot,np.zeros(int(len(x_values)-len(pot))))
    return pot

maxTime = 22/k0
# time points
t = np.linspace(0,maxTime,1000000)
"""discretised initial wave packet"""
wavePacket = psi_t0(x_values,k0)
probabilityDistribution = abs(wavePacket)**2
# concatanate real and imaginary components
wavefunction_concatanated = np.concatenate((wavePacket.real,wavePacket.imag), axis=None)

"""matrix of second order position derivative"""
M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()

"""create figure"""
fig, ax = plt.subplots(2,2)
ax[0][0].set_xlabel('x')
ax[0][0].set_ylabel('probability')
ax[0][0].set_xlim(-25,25)
ax[0][0].set_ylim(0,0.21)

ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('probability')
ax[1][0].set_xlim(-25,25)
ax[1][0].set_ylim(0,0.21)

ax[0][1].set_xlabel('x')
ax[0][1].set_ylabel('probability')
ax[0][1].set_xlim(-25,25)
ax[0][1].set_ylim(0,0.21)

ax[1][1].set_xlabel('x')
ax[1][1].set_ylabel('probability')
ax[1][1].set_xlim(-25,25)
ax[1][1].set_ylim(0,0.21)

V0_vals = np.array([1,2])
a_vals = np.array([5,10])
start_time = time.time()
for i in range(0,2):
    V0 = V0_vals[i]
    for i2 in range(0,2):
        V = np.diag(np.full(N_discretised,potential(x_values,a_vals[i],V0_vals[i2])))
        
        """solving ODE"""
        
        G = (-hBar**2)/(2*m)*M + V
        
        # block matrix
        B = np.block([[np.zeros((N_discretised,N_discretised)),G],[-G,np.zeros((N_discretised,N_discretised))]])
        
        # function that returns dpsi/dt
        def rhs(psi,t):
            return 1/hBar*(np.matmul(B,psi))
        
        
        # solve ODE
        all_wavefunction_conc = odeint(rhs, wavefunction_concatanated, t)
        final_wavefunction_conc = all_wavefunction_conc[-1]
        final_wavefunction_conc_split = np.split(final_wavefunction_conc,2)
        final_wavefunction = final_wavefunction_conc_split[0]+1j*final_wavefunction_conc_split[1]
        
        """discretised final wave packet"""
        probabilityDistribution_final = abs(final_wavefunction)**2
        
        ax[i][i2].set_title('V0='+str(round(V0_vals[i2],2))+' width='+str(round(a_vals[i]*2*width/N_discretised,2)))
        ax[i][i2].plot(x_values,probabilityDistribution,'r',label='initial')
        ax[i][i2].plot(x_values,probabilityDistribution_final,label='final')
        ax[i][i2].vlines(0,-10,10,'b','dashed')
        ax[i][i2].legend(loc='upper center',title_fontsize='large')
        ax[i][i2].text(-22,0.175,'Reflected='+str(round(np.sum(probabilityDistribution_final[0:int(len(probabilityDistribution_final)/2+0.5)])/(N_discretised/50),4)),backgroundcolor='palegreen')
        ax[i][i2].text(10,0.175,'Transmitted'+str(round(np.sum(probabilityDistribution_final[int(len(probabilityDistribution_final)/2+0.5+N_discretised/(2*width)):int(len(probabilityDistribution_final))])/(N_discretised/50),4)),backgroundcolor='palegreen')
    
    end_time = time.time()
    print('Elapsed time = ', repr(end_time - start_time))
