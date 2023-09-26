Quantum Tunneling and Reflection

This project aims to look at the behaviour of a particle (propagating Gaussian wave packet) hitting a potential barrier.
The potentials looked at are the step-like potential for different widths and heights and a reflectionless potential.
The project also shows the limitations of the simulation and best practices while using the programs.

Modules used:
matplotlib
numpy
time
scipy

Features:
A common problem of the program was that the particles would often hit the edges of the simulation. To combat this, the maximum time allowed is computed using the initial energy of the wave packet, so that the distance travelled by the wave is always constant.

How to run the project:
To run the core routine (test_case_V0), you will only need to click run and the relevant plots will be produced automatically
All other codes run in the same way, these are the relevant programs for the figures
However, the input variable can be edited quite easily in the 'initial values' section of code.
The potential can be edited in the potential(x) function to produce any potential, just parse in x_values into whatever equation you want the potential function to return.

Units:
All values produced from code must be put through a scale factor to reproduce real-world results.
Say you are looking at things on the nanoscale, the length scale would be 1e-9m
Energy scale is equal to the hBar^2/(m*(length_scale)^2), where m is the effective mass, and hBar = 1.05457182e-34 m^2kg/s
Time scale is equal to hBar/(energy_scale)
So for Gallium Arsenide; length_scale = 1e-9m, effective_mass = 9.11e-31*0.067kg, epsilon_scale = 1.1eV, time_scale = 5.8e-16s