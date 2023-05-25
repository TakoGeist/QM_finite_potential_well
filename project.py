
"""
    File: projeto.py
    Summary: Implementation of a solver for the Schrödinger time-independent
             equation for a finite potential well using shooting's method 
             and the standard approximation of the 2nd order derivative.
             The program takes as environment variables the well's width in 
             angstrongs, the potential of the well in eletron volts and 
             wether or not to solve it taking it as an eigenvalue problem, 
             this last one is selected with a 'y'.
             Not providing these variables makes them default to 
             10 angstrongs, 10 eV and 'n', respectively.
    Authors: Diogo Ramos, a95109, Universidade do Minho
             Gabriel Costa, a94893, Universidade do Minho
             Pedro Martins, a95445, Universidade do Minho
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from scipy.linalg import eigh_tridiagonal
plt.style.use('dark_background')

def calculate_wave(m, energy, init, well, store = 'n', wave = []):
    """ Schrödinger time-independent equation solver using standard approximation for 2nd order derivatives 

    Args:
        m (any): Particle's mass
        energy (any): Energy level to use in calculation
        init (any): Tuple with initial conditions
        well (any): Potential well; Should expand to a domain where 0 is an expected solution
        store (str, optional): Wether or not to store the values of the eigenfunction. Defaults to 'n'.
        wave (list, optional): List to store the values of the eigenfunction. Defaults to [].

    Returns:
        any: Eigenfunction value at the end of the given potential well
    """
    hbar = 1.05457182e-34
    temp = []
    phi0 = init[0]
    phi1 = init[1]
    for pot in well:
        phi2 = ((pot - energy) * (dz**2 * 2 * m) / hbar**2 + 2) * phi1 - phi0
        phi0 = phi1
        phi1 = phi2
        temp.append(phi1)
    if store == 'y':
        wave.append(temp)
    return phi1

def find_energy(m, well, final_energy, parity):
    """Schrödinger time-independent equation solver for energy levels of bound-states  

    Args:
        m (any): Particle's mass
        well (any): Potential Well
        final_energy (list): Array to return energies
        parity (tuple(any,any)): Initial conditions for even and odd solutions
    """
    energy = np.linspace(V/len(well), V, len(well)-1)
    de = V / N

    for (par,init) in enumerate(parity):
        old_phi = calculate_wave(m, 0, init, well)
        for (i,e) in enumerate(energy):
            phi1 = calculate_wave(m, e, init, well)
            if old_phi * phi1 < 0:
                final_energy.append((par,(np.abs(old_phi) / (np.abs(phi1) + np.abs(old_phi))) * de + energy[i] - de))
            old_phi = phi1

    final_energy.sort(key= lambda x: x[1])

def eigen_method(m, N, dz, well, V):
    """Schrödinger time-independent equation solver taking the equation as a eigenvalue problem

    Args:
        m (any): Particle's mass
        N (any): Number of sample points
        dz (any): Distance Step
        Well (any): Potential Well
        V (any): Potential
    """
    hbar = 1.05457182e-34
    final_energy = []
    wave = []

    diagonal =  np.ones(N) * hbar ** 2 / (2 * m) * (2 / dz ** 2) + well
    remainder = - np.ones(N - 1) * hbar ** 2 / (2 * m) / dz ** 2

    final_energy, wave = eigh_tridiagonal(diagonal, remainder, select='v', select_range=(0, V))

    return (np.array(final_energy), np.transpose(np.array(wave)))
    

if __name__ == "__main__":
    L = 10e-10              #Default well width
    V = 10 * 1.602e-19      #Default potential value
    real = False            #Default flag

    if len(argv) > 1:
        L = int(argv[1]) * 1e-10
    if len(argv) > 2:
        V = int(argv[2]) * 1.602e-19
    if len(argv) > 3:
        real = argv[3] == "y"
    
    hbar = 1.05457182e-34
    m = 9.109e-31           #Particle's mass
    N = 2000                #Resolution to be used on energy calculations
    z = np.linspace(0, 3 * L / 4, N)
    neg_z = np.linspace(0, -3 * L / 4, N)
    dz = (3 * L / 4) / N
    well = V - V * (- L / 2 <= z) * (z <= L / 2)
    neg_well = V - V * (- L / 2 <= neg_z) * (neg_z <= L / 2)

    colors = ("turquoise", "darkorange", "fuchsia", "dodgerblue", "chartreuse", "darkorchid", "aqua",
              "deepskyblue", "limegreen", "mediumslateblue", "orangered", "khaki", "royalblue")

    final_energy = []
    parity = [(dz, dz), (0, dz)]
    
    find_energy(m, well, final_energy, parity)

    fig, ax = plt.subplots(figsize=(12,9))
    color_set = iter(colors)

    for i in range(0, len(final_energy)):

        try:
            paint = next(color_set)
        except StopIteration:
            color_set = iter(colors)
            paint = next(color_set)

        ax.hlines(final_energy[i][1]/1.602e-19, 0, 1, linewidth=1.3, color=paint)
        ax.text(1, final_energy[i][1]/1.602e-19 - 0.01*V/1.602e-19, 
            "$E_{%i}" %(i+1) + r" = %.3f$ eV"%(final_energy[i][1]/1.602e-19), fontsize=13, color=paint)

    ax.hlines(V/1.602e-19, 0, 1, color="crimson")
    ax.text(0.9, 1.02 * V/1.602e-19, "V = %.3f eV"%(V/1.602e-19), fontsize=13, color="crimson")
    ax.axes.get_xaxis().set_visible(False)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_title("Energy Levels", fontsize=32)
    ax.set_ylabel("Energy (eV)", fontsize=20)
    ax.yaxis.tick_left()
    ax.axis([0, 1, 0, V/1.602e-19*1.1])
    plt.show()

    fig, ax = plt.subplots(figsize=(12,9))
    wave = []
    neg_wave = []
    for (par, val) in final_energy:
        if par == 0:
            calculate_wave(m, val, (dz, dz), well, store='y', wave=wave)
            calculate_wave(m, val, (dz, dz), neg_well, store='y', wave=neg_wave)
        else:
            calculate_wave(m, val, (0, dz), well, store='y', wave=wave)
            calculate_wave(m, val, (0, -dz), well, store='y', wave=neg_wave)

    
    final_energy = np.array([x[1] for x in final_energy])
    final_energy = final_energy / 1.602e-19

    for i in range(0,len(wave)):
        wave[i] = list(reversed(neg_wave[i])) + wave[i]
    wave = np.array(wave)
    wave = wave / 1.602e-19

    ax.axes.get_yaxis().set_visible(False)
    
    a = np.linspace(-3 * L / 4, 3 * L / 4, 2*N)
    color_set = iter(colors)

    scale = 0
    scale = 2 * np.sqrt(final_energy[len(final_energy)-1] / final_energy[0])

    for i in range(0,len(wave)):

        try:
            paint = next(color_set)
        except StopIteration:
            color_set = iter(colors)
            paint = next(color_set)

        norm = np.sqrt(sum(wave[i]**2))

        ax.plot(a/1e-10,final_energy[i] + wave[i]/norm * scale, color=paint)

        ax.hlines(final_energy[i], -3 * L/1e-10 / 4, 3 * L/1e-10 / 4, linewidth=1.3, linestyle='--', color="gray")

        ax.text(3 * L/1e-10 / 4, final_energy[i] - 0.01*V/1.602e-19, 
            "$E_{%i}" %(i+1) + r" = %.3f$ eV"%(final_energy[i]), fontsize=13, color=paint)

    ax.plot(z/1e-10, well/1.602e-19,color= "crimson")
    ax.plot(neg_z/1e-10, neg_well/1.602e-19,color= "crimson")
    ax.text(8 * L/1e-10 / 15, 1.02 * V/1.602e-19, "V = %.3f eV"%(V/1.602e-19), fontsize=13, color="crimson")
    ax.margins(0.00)
    ax.axis([-3 * L/1e-10 / 4, 3 * L/1e-10 / 4,0.0,1.1* V/1.602e-19])
    
    ax.annotate('L/2', xy=(L/1e-10 / 2, 0), xytext=(L/1e-10 / 2 - 0.1, -0.3), fontsize=12)
    ax.annotate('-L/2', xy=(-L/1e-10 / 2, 0), xytext=(-L/1e-10 / 2 - 0.2, -0.3), fontsize=12)
    ax.set_title("Funções de onda", fontsize=32)
    ax.set_xlabel("Distance (\u212B)", fontsize=20)

    plt.show()

    """Generate Solutions treating the problem as an eigenvalue problem, Solving using a tridiagonal matrix obtained from
        the standard approximation of the 2nd order derivative
    """
    if real:
        N = 50000
        z = np.linspace(-3*L / 4, 3 * L / 4, N)
        dz = (3/2 * L) / N

        real_energy, real_wave = eigen_method(m, N, dz, V - V * (- L / 2 <= z) * (z <= L / 2), V)

        real_wave = real_wave / 1.602e-19

        for i in range(0,len(real_energy)):
            norm = np.sqrt(sum(real_wave[i,:]**2))
            real_wave[i] = real_wave[i] / norm

        fig, ax = plt.subplots(figsize=(12,9))

        color_set = iter(colors)
        for i in range(0,len(real_energy)):

            try:
                paint = next(color_set)
            except StopIteration:
                color_set = iter(colors)
                paint = next(color_set)

            ax.plot(z/1e-10,real_energy[i]/1.602e-19 + real_wave[i] * 50, color=paint)
            ax.hlines(real_energy[i]/1.602e-19, -3 * L/1e-10 / 4, 3 * L/1e-10 / 4, linewidth=1.3, linestyle='--', color="gray")
            ax.text(3 * L/1e-10 / 4, real_energy[i]/1.602e-19 - 0.01*V/1.602e-19, 
                "$E_{%i}" %(i+1) + r" = %.3f$ eV"%(real_energy[i]/1.602e-19), fontsize=13, color=paint)


        ax.plot(z/1e-10, (V - V * (- L / 2 <= z) * (z <= L / 2))/1.602e-19,color= "crimson")
        ax.text(8 * L/1e-10 / 15, 1.02 * V/1.602e-19, "V = %.3f eV"%(V/1.602e-19), fontsize=13, color="crimson")
        ax.margins(0.00)
        ax.axis([-3 * L/1e-10 / 4, 3 * L/1e-10 / 4,0.0,1.1* V/1.602e-19])

        ax.annotate('L/2', xy=(L/1e-10 / 2, 0), xytext=(L/1e-10 / 2 - 0.1, -0.3), fontsize=12)
        ax.annotate('-L/2', xy=(-L/1e-10 / 2, 0), xytext=(-L/1e-10 / 2 - 0.2, -0.3), fontsize=12)
        ax.set_title("Funções de onda", fontsize=32)
        ax.set_xlabel("Distance (\u212B)", fontsize=20)

        plt.show()