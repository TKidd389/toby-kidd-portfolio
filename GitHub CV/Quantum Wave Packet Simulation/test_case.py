# -*- coding: utf-8 -*-
"""
@author: tobyk
"""


"""import modules"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags # matrix of second order position derivative
from scipy.integrate import odeint
import time # times how long code takes to run

"""close pre-existing plots"""
plt.close('all')

"""inital values"""
x0 = -10 # centre of wave packet
sigma = 2 # width of wave packet

N_discretised = 501 # number of x values computed
width = 25 # width of total allowed x values

V0 = None # potential height
E = 2 # initial energy of wavepacket
hBar = 1
m = 1
k0 = np.sqrt(2*m*E)/hBar
def psi_t0(x):
    return np.exp(-(x-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x)


"""create figure"""
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('position [a.u]')
ax1.set_ylabel('probability')
ax1.title.set_text('Free Wave Packet')
ax1.set_xlim(-width,width)
ax1.set_ylim(0,0.21)

"""discretised initial wave packet"""
x_values = np.linspace(-width,width,N_discretised)
wavePacket = psi_t0(x_values)
probabilityDistribution = abs(wavePacket)**2

"""potential as a matrix"""
def potential(x):
    return np.zeros(len(x_values))
V = np.diag(np.full(N_discretised,potential(x_values)))

"""plot potential"""
ax1.plot(x_values, potential(x_values))

"""plot wave packets"""
ax1.plot(x_values, probabilityDistribution, 'r.', label='t=0 (numeric solution)')

"""matrix of second order position derivative"""
M = diags([np.ones(N_discretised-1)/(2*width/N_discretised)**2,-2*np.ones(N_discretised)/(2*width/N_discretised)**2,np.ones(N_discretised-1)/(2*width/N_discretised)**2],[-1,0,1]).toarray()

"""solving ODE"""
# time points
maxTime = 10
t = np.linspace(0,maxTime,1000000)

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

"""plotting at final time"""
ax1.plot(x_values, probabilityDistribution_final, 'b.', label='t=tend (numeric solution)')

end_time = time.time()

"""theoretical results"""
x_values_theory = np.linspace(-width,width,width*100)
sigma_t = sigma*(1+maxTime**2/4/sigma**4)**0.5
# t=0
ax1.plot(x_values_theory,abs(np.exp(-(x_values_theory-x0)**2/(4*sigma**2))/((2*np.pi)**0.25*np.sqrt(sigma))*np.exp(1j*k0*x_values_theory))**2, 'r', label='t=0 (theory)')
# t=tend
ax1.plot(x_values_theory,(np.sqrt(2*np.pi)*sigma_t)**(-1)*np.exp(-(x_values_theory-x0-k0*hBar/m*maxTime)**2/2/sigma_t**2), 'black', label='t=tend(theory)')

"""legend"""
ax1.legend(loc='upper right',title_fontsize='xx-large')

print('Elapsed time = ', repr(end_time - start_time))

"""
Values used for calculations in later programs
Important for ensuring that returned values have conserved quantities
, in case any errors do occur
"""
transmitted = np.sum(probabilityDistribution_final[int(len(probabilityDistribution_final)/2+0.5+N_discretised/(2*width)):int(len(probabilityDistribution_final))])/(501/50)
reflected = np.sum(probabilityDistribution_final[0:int(len(probabilityDistribution_final)/2+0.5)])/(501/50)

# potential energy values
exp_V1 = np.matmul(np.conj(psi_t0(x_values)),np.matmul(V,psi_t0(x_values)))
exp_V2 = np.matmul(np.conj(final_wavefunction),np.matmul(V,final_wavefunction))
# kinetic energy values
exp_T1 = -np.matmul(np.conj(psi_t0(x_values)),np.matmul(M,psi_t0(x_values)))/2
exp_T2 = -np.matmul(np.conj(final_wavefunction),np.matmul(M,final_wavefunction))/2
# total energy values
exp_E1 = exp_V1 + exp_T1
exp_E2 = exp_V2 + exp_T2
