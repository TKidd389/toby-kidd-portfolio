# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 08:53:18 2022

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
def psi_t0(x,k0,sigma):
    return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)

V0 = 24.5
"""potential as a matrix"""
def potential(x,a,V0):
    pot = np.append(np.zeros(int((N_discretised-1)/2)),np.ones(a)*V0)
    pot = np.append(pot,np.zeros(int(len(x_values)-len(pot))))
    return pot

maxTime = 22/k0
# time points
t = np.linspace(0,maxTime,1000)

"""matrix of second order position derivative"""
M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()

"""create figure"""
fig, ax = plt.subplots(2,2)
ax[0][0].set_title('Simulated')
ax[0][0].set_xlabel('Initial '+r'$\sigma$')
ax[0][0].set_ylabel('-Final Height')

ax[1][0].set_title('Theoretical')
ax[1][0].set_xlabel('Initial '+r'$\sigma$')
ax[1][0].set_ylabel('Final '+r'$\sigma$')

ax[0][1].set_title('Simulated')
ax[0][1].set_xlabel('Initial '+r'$\sigma$')
ax[0][1].set_ylabel('-Final Height')

ax[1][1].set_title('Theoretical')
ax[1][1].set_xlabel('Initial '+r'$\sigma$')
ax[1][1].set_ylabel('Final '+r'$\sigma$')


sigma_vals = np.arange(0,10,0.1)
final_height_vals = np.array([])
start_time = time.time()
for i in range(0,len(sigma_vals)):
    """discretised initial wave packet"""
    wavePacket = psi_t0(x_values,k0,sigma_vals[i])
    probabilityDistribution = abs(wavePacket)**2
    # concatanate real and imaginary components
    wavefunction_concatanated = np.concatenate((wavePacket.real,wavePacket.imag), axis=None)
    
    V = np.diag(np.full(N_discretised,potential(x_values,0,0)))
    
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
    

    final_height_vals = np.append(final_height_vals,probabilityDistribution_final[np.argmax(probabilityDistribution_final)])

    end_time = time.time()
    #print('Elapsed time = ', repr(end_time - start_time))
    

def sigma_t(sigma):
    return sigma*(1+maxTime**2/4/sigma**4)**0.5
ax[1][0].plot(sigma_vals, sigma_t(sigma_vals))
ax[1][0].axvline(x=1.4, color='black', linestyle='dashed')
ax[1][0].axvline(x=3, color='black', linestyle='dashed')

ax[0][0].plot(sigma_vals, -final_height_vals, 'b*-')
ax[0][0].axvline(x=1.4, color='black', linestyle='dashed')
ax[0][0].axvline(x=3, color='black', linestyle='dashed')

ax[0][1].plot(sigma_vals[14:30], -final_height_vals[14:30], 'b*-')

ax[1][1].plot(sigma_vals[14:30], sigma_t(sigma_vals[14:30]))