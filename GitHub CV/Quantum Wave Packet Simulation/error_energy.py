# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:50:45 2022

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

V0 = 0
E = 2
hBar = 1
m = 1
k0 = np.sqrt(2*m*E)/hBar
def psi_t0(x):
    return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)


"""create figure"""
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time')
ax1.set_ylabel('Energy')

"""discretised initial wave packet"""
x_values = np.linspace(-width,width,N_discretised)
wavePacket = psi_t0(x_values)
probabilityDistribution = abs(wavePacket)**2

"""potential as a matrix"""
def potential(x):
    return x**2
V = np.diag(np.full(N_discretised,potential(x_values)))

"""matrix of second order position derivative"""
M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()

"""solving ODE"""
# time points
maxTime = 22/k0
timeSteps = 1000
t = np.linspace(0,maxTime,timeSteps)

# concatanate real and imaginary components
wavefunction_concatanated = np.concatenate((wavePacket.real,wavePacket.imag), axis=None)

G = (-hBar**2)/(2*m)*M + V

# block matrix
B = np.block([[np.zeros((N_discretised,N_discretised)),G],[-G,np.zeros((N_discretised,N_discretised))]])

# function that returns dpsi/dt
def rhs(psi,t):
    return 1/hBar*(np.matmul(B,psi))

exp_V1 = np.matmul(np.conj(psi_t0(x_values)),np.matmul(V,psi_t0(x_values))) *(2*width/N_discretised)
exp_T1 = -np.matmul(np.conj(psi_t0(x_values)),np.matmul(M,psi_t0(x_values)))/2 *(2*width/N_discretised)
exp_E1 = exp_V1 + exp_T1

start_time = time.time()
# solve ODE
all_wavefunction_conc = odeint(rhs, wavefunction_concatanated, t)
pot_energy = np.array([])
kin_energy = np.array([])
tot_energy = np.array([])
for i in range(0,len(all_wavefunction_conc)):
    final_wavefunction_conc = all_wavefunction_conc[i]
    final_wavefunction_conc_split = np.split(final_wavefunction_conc,2)
    final_wavefunction = final_wavefunction_conc_split[0]+1j*final_wavefunction_conc_split[1]
    exp_T2 = -np.matmul(np.conj(final_wavefunction),np.matmul(M,final_wavefunction))/2 *(2*width/N_discretised)
    exp_V2 = np.matmul(np.conj(final_wavefunction),np.matmul(V,final_wavefunction)) *(2*width/N_discretised)
    exp_E2 = exp_V2 + exp_T2
    pot_energy = np.append(pot_energy, exp_V2)
    kin_energy = np.append(kin_energy, exp_T2)
    tot_energy = np.append(tot_energy, exp_E2)

"""plotting energies"""
ax1.plot(t, tot_energy, 'b.-', label='Total Energy')
ax1.plot(t, kin_energy, 'r.-', label='Kinetic Energy')
ax1.plot(t, pot_energy, 'g.-', label='Potential Energy')


"""legend"""
ax1.legend(loc='lower right',title_fontsize='xx-large')

end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))