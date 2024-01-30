# 2D Ising Model Simulation

## Introduction

This repository contains Python code for simulating the 2D Ising model using the Metropolis algorithm. The Ising model is a mathematical model of ferromagnetism in statistical mechanics, representing spins on a lattice that interact with each other. The Metropolis algorithm is a Monte Carlo method used to sample from probability distributions.

## Getting Started

To run the simulation, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required dependencies by running:
```bash
pip install numpy matplotlib numba scipy
```
4. Run the script `ising_model_simulation.py`:

```bash
python ising_model_simulation.py
```

## Code Structure
The code consists of the following main parts:

initial_state(N): Initializes the lattice with random spins.

hamiltonian(r, c, lattice, N): Calculates the Hamiltonian of a spin.

checkflip(r, c, lattice, T, N): Checks if a spin should flip based on the Metropolis algorithm.

magnetization(lattice): Calculates the magnetization of the lattice.

energy(lattice): Calculates the total energy of the lattice.

montecarlo(lattice, T, eqsteps): Performs Monte Carlo moves using the Metropolis algorithm.

## Parameters
N: Size of the lattice.

nt: Number of temperature points.

eqSteps: Number of Monte Carlo moves for equilibration.

mcSteps: Number of Monte Carlo moves for calculation.

T: Array of temperature points.

## Visualization
The simulation results are visualized using Matplotlib. The plots include:

Average magnetization per spin vs. temperature.

Average energy per spin vs. temperature.

Specific heat vs. temperature.

Magnetic susceptibility vs. temperature.

Additionally, the critical exponents of the Ising model are calculated and plotted.

## Output
The simulation generates plots of the simulation results and critical exponents, saved as PNG files.
