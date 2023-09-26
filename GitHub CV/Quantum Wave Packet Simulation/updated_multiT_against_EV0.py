# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:45:12 2022

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

maxTime = 6.5
x_values = np.linspace(-width,width,N_discretised)

E = 6
hBar = 1
m = 1
k0 = np.sqrt(2*m*E)/hBar
def psi_t0(x,k0):
    return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)

V0 = 24.5
"""potential as a matrix"""
def potential(x):
    a = np.copy(x)
    a[a<0] = 0
    a[a>1]=0
    a = np.where(a != 0, V0, a)
    return a

V0_vals = np.array([1,4.5,24.5])
E_vals = np.arange(0.1,3.1,0.03)
T_vals = np.array([])
start_time = time.time()
for i in range(0,3):
    V0 = V0_vals[i]
    for i in range(0,len(E_vals)):
        k0 = np.sqrt(2*m*E_vals[i]*V0)/hBar
        maxTime = 22/k0
        """discretised initial wave packet"""
        wavePacket = psi_t0(x_values,k0)
        probabilityDistribution = abs(wavePacket)**2
        
        
        V = np.diag(np.full(N_discretised,potential(x_values)))
        
        
        """matrix of second order position derivative"""
        M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()
        
        """solving ODE"""
        # time points
        t = np.linspace(0,maxTime)
        
        # concatanate real and imaginary components
        wavefunction_concatanated = np.concatenate((wavePacket.real,wavePacket.imag), axis=None)
        
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
        
        T_vals = np.append(T_vals,np.sum(probabilityDistribution_final[int(len(probabilityDistribution_final)/2+0.5+N_discretised/(2*width)):int(len(probabilityDistribution_final))])/(N_discretised/50))
    
    end_time = time.time()
    print('Elapsed time = ', repr(end_time - start_time))

T_vals_split = np.split(T_vals,3)

"""plot"""
"""create figure"""
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('E/V0')
ax1.set_ylabel('T')
ax1.set_xlim(0.2,3)
ax1.set_ylim(0,1.1)
ax1.set_xticks([0.2,0.5,1.0,1.5,2.0,2.5,3.0])

ax1.plot(E_vals,T_vals_split[0],'r',label='1') # E_vals = E/V0 because V=1
ax1.plot(E_vals,T_vals_split[1],'g',label='3')
ax1.plot(E_vals,T_vals_split[2],'b',label='7')
classical = np.append(np.zeros(int(len(E_vals)/3)-2),np.ones(int(len(E_vals)/3*2)+3))
ax1.step(E_vals,classical,linestyle='dashed',where='pre',label='classical')

ax1.legend(title=r'$\sqrt{2mV)}a/\hbar=$',loc='lower right',title_fontsize='xx-large')