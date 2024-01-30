'''
Simulation of the 2D Ising model using the Metropolis algorithm
by Mohaddeseh Mozaffari
Shahid Beheshti University
'''
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import curve_fit

#generates a random square lattice
def initial_state(N):
    lattice = np.random.choice([-1,1],(N,N))
    return lattice

#calculate the hamiltonian of a spin
@njit
def hamiltonian(r, c, lattice, N):

    def bc(i):

        if i > N-1:
            return 0
        if i < 0:
            return N-1
        else:
            return i

    E = - lattice[r][c]* (lattice[bc(r+1)][c]+ lattice[bc(r-1)][c]+
                             lattice[r][bc(c-1)]+ lattice[r][bc(c+1)])
    return E

#check spin is flip or not
@njit
def checkflip(r, c, lattice, T, N):

    dE = -2* hamiltonian(r,c,lattice,N)

    if  dE <= 0:
        lattice[r][c] *= -1

    elif np.random.rand() < np.exp(-dE/T):
        lattice[r][c] *= -1

    return lattice

#calculate magnetization of the lattice
def magnetization(lattice):
    mag = np.sum(lattice)
    return mag

#calculate total energy of the lattice
@njit
def energy(lattice):
    n = len(lattice)
    E = 0
    for r in range(n):
        for c in range(n):
            E += hamiltonian(r,c,lattice,n)
    return E/2

#Monte Carlo move using Metropolis algorithm
@njit
def montecarlo(lattice, T, eqsteps):
    for _ in range(0, N**2):
                        
        row = np.random.randint(0, N)
        col = np.random.randint(0, N) 
        lattice = checkflip(row, col, lattice, T, N)

    return lattice

    

N = 16 #size of the lattice
nt = 40 #number of temperature points
eqSteps = 2000 #number of Monte Carlo moves for equilibration
mcSteps = 1000  #number of Monte Carlo moves for calculation

T = np.linspace(1, 4, nt)
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)

n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 


#main program
for tt in range(nt):
    E1 = M1 = E2 = M2 = 0
    config = initial_state(N)
    t = T[tt]
    
    for i in range(eqSteps):         
        montecarlo(config, t, N)
        currentMag = abs(magnetization(config))/float(N**2)
                  
        if currentMag == 1: 
             break          

    for i in range(mcSteps):
        montecarlo(config, t, N)           
        Ene = energy(config)     
        Mag = magnetization(config)     

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene

    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)/(t*t)
    X[tt] = (n1*M2 - n2*M1*M1)/t

fig, axs = plt.subplots(2, 2, figsize= (10,8))
fig.tight_layout(pad=4.0)

for ax in axs.flatten():
    ax.tick_params(color='#708090', labelcolor='#708090')
    for spine in ax.spines.values():
        spine.set_edgecolor('#708090')

fig.suptitle(f'Ising model for N = {N}', fontsize=16)

line1 = axs[0,0].plot(T, abs(M), marker='o', color='#661D98',ls='None')
axs[0,0].set(xlabel= 'Temperature', ylabel= 'Average Magnetisation per Spin')
axs[0,0].set_title('Average Magnetisation vs Temperature')


line2 = axs[0,1].plot(T, E, marker='o', color='#2C6FFE',ls='None')
axs[0,1].set(xlabel= 'Temperature ', ylabel= 'Average Energy per Spin')
axs[0,1].set_title('Average Energy vs Temperature')


line3 = axs[1,0].plot(T, C, 'og')
axs[1,0].set(xlabel= 'Temperature', ylabel= 'Specific Heat')
axs[1,0].set_title('Specific Heat vs Temperature')


line4 = axs[1,1].plot(T, X, 'oy')
axs[1,1].set(xlabel= 'Temperature', ylabel= 'Susceptibility')
axs[1,1].set_title('Magnetic Susceptibility vs Temperature')

plt.savefig(f'Ising_{N}.png', dpi=300)

TT = np.log(T)[15:17]
MM = np.log(abs(M))[15:17]
CC = np.log(C)[15:17]
XX = np.log(X)[15:17]

def line(x,a,b):
    return a*x+b

param, _ = curve_fit(line,TT, MM)
parac, _ = curve_fit(line,TT, CC)
parax, _ = curve_fit(line,TT, XX)

fig, axs = plt.subplots(1, 3, figsize= (12,4))
fig.tight_layout(pad=4.0)

for ax in axs.flatten():
    ax.tick_params(color='#708090', labelcolor='#708090')
    for spine in ax.spines.values():
        spine.set_edgecolor('#708090')

fig.suptitle(f'Critical exponents of Ising model for N = {N}', fontsize=16)

line1 = axs[0].plot(TT, MM, marker='o', color='#661D98',ls='None')
line2 = axs[0].plot(TT, param[0]*TT+param[1])
axs[0].set(xlabel= r'$\ln(T)$', ylabel= r'$\ln(M)$')
axs[0].set_title(fr'$\alpha =${param[0]}')

line3 = axs[1].plot(TT, CC, 'og')
line4 = axs[1].plot(TT, parac[0]*TT+parac[1])

axs[1].set(xlabel= r'$\ln(T)$', ylabel= r'$\ln(C)$')
axs[1].set_title(fr'$\beta =${parac[0]}')


line5 = axs[2].plot(TT, XX, 'oy')
line6 = axs[2].plot(TT, parax[0]*TT+parax[1])

axs[2].set(xlabel= r'$\ln(T)$', ylabel= r'$\ln( \chi)$')
axs[2].set_title(fr'$\gamma =${parax[0]}')

plt.savefig(f"critical exponent_{N}", dpi=300)


