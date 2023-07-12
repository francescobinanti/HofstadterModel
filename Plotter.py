import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import GenericModule as gm
    
# System parameters
J = 1.
U = 0.
U3 = 0.
FluxDensity = 0.2
trapConf = 0.

# Handle the parameters parsed by the command line
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, help='number of particles')
parser.add_argument('-L', type=int, help='side of the square lattice of size LxL')
parser.add_argument('-J', type=float, help='tunneling energy')
parser.add_argument('-U', type=float, help='two-body onsite interaction (only in softcore mode)')
parser.add_argument('-U3', type=float, help='three-body onsite interaction (only in softcore mode)')
parser.add_argument('-r0', type=float, help='Laguerre-Gauss radius parameter (for a pure gaussian beam it is the gaussian dispersion)')
parser.add_argument('--conf', type=float, help='harmonic trap confinement strength (v0) as v0 * r^2')
parser.add_argument('--gamma', type=float, default=2, help='trap steepness (g) as v0 * (r)^g (default=2)')
parser.add_argument('--alpha', type=float, help='magnetic flux density as alpha=p/q')
parser.add_argument('--hardcore', type=int, nargs='?', const=1, default=0, help='hardcore bosons mode')
# Plotting parameters
parser.add_argument('--spectrumflux', type=int, nargs='?', const=1, default=0, help='plot the energy spectrum E vs the magnetic flux density alpha')
parser.add_argument('--alphainit', type=float, default=0.01, help='initial value of the flux alpha for the energy plot E vs alpha')
parser.add_argument('--alphafinal', type=float, default=0.1, help='final value of the flux alpha for the energy plot E vs alpha')
parser.add_argument('--alphastep', type=float, default=0.01, help='step for the increase of alpha in the E vs alpha plot')
parser.add_argument('--c4spectrum', type=int, nargs='?', const=1, default=0, help='plot the energy spectrum E vs C4 quantum numbers')
parser.add_argument('--density', type=int, help='plot the local density of |psi_n> on the square lattice (provide n as parameter)')
parser.add_argument('--absorption', type=int, nargs='?', const=1, default=0, help='plot the absorption spectrum from data file generated by LaguerreGaussAbsorption.py')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U3 = args.U
if args.U3 is not None: U3 = args.U3
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf

gamma = args.gamma

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True
    
if args.spectrumflux == 0:
    spectrumFlux = False
elif args.spectrumflux == 1:
    spectrumFlux = True
    
if args.c4spectrum == 0:
    c4Spectrum = False
elif args.c4spectrum == 1:
    c4Spectrum = True
    
alphaInit = args.alphainit
alphaFinal = args.alphafinal
alphaStep = args.alphastep

Ns = L*L

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True

if (spectrumFlux == True):
    for a in np.arange(alphaInit, alphaFinal+alphaStep, alphaStep):
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=round(a,2), N=N, spectrum=True)
        energies = gm.LoadSpectrum(fileName)
        
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$E-E_0$')
        plt.scatter(np.full(len(energies), a), energies-energies[0], marker='o', s=1.5, facecolors='#0066cc')
    plt.savefig(fileName+'_plot.pdf')
    
if (c4Spectrum == True):
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, spectrum=True, c4=True)
    data = gm.LoadC4Spectrum(fileName)
    
    fig, ax = plt.subplots()
    gsEnergy = np.min(data[1])
    
    for i in np.arange(0,len(data[0])):
        rect = patches.Rectangle((data[0][i] - 0.125, (data[1][i]-gsEnergy)), width=0.25, height=0.003, facecolor='blue')
        ax.add_patch(rect)
            
    ax.set_xlim(min(data[0]) - 0.5, max(data[0]) + 0.5)
    ax.set_ylim(min(data[1]) - gsEnergy - 0.1, max(data[1]) - gsEnergy + 0.1)
    
    ax.set_xticks([0,1,2,3])
    
    plt.xlabel(r'$L_4$')
    plt.ylabel(r'$E-E_0$')
    
    plt.savefig(fileName+'_plot.pdf')
    
if args.density is not None:
    selEigenstateDen = args.density
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, selEigenstateDen, U3=U3, alpha=FluxDensity, N=N, localDensity=True)
    x, y, density = gm.LoadDensity(fileName)
    x = x.astype(int)
    y = y.astype(int)
    densityArray = np.zeros((L,L))
    densityArray[x,y] = density
    plt.imshow(densityArray, cmap='inferno', extent=[-0.5, L - 0.5, -0.5, L - 0.5])
    plt.colorbar(label=r'$\left < \hat n_{xy} \right >$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.xticks(np.arange(L), np.arange(L))
    plt.yticks(np.arange(L), np.arange(L))
    plt.savefig(fileName+'.pdf', format='pdf')
    
if args.absorption == 1:
    fig, ax = plt.subplots()
    cmap = mpl.colormaps['ocean_r']
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=0.7))
    #norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    
    # l < 0 region
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, absSpectrum=True, r0=args.r0)
    energy, l, matElem = gm.LoadFileThree(fileName+'_neg')
    
    matElem[matElem < 1e-5] = np.NaN
    
    sm.set_array(matElem)
    colors = sm.to_rgba(matElem)
    
    for i in np.arange(0,len(energy)):
        rect = patches.Rectangle((l[i] - 0.45, energy[i]), width=0.9, height=0.003, color=colors[i])
        ax.set_facecolor(cmap(0))
        ax.add_patch(rect)
        
    # l > 0 region
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, absSpectrum=True, r0=args.r0)
    energy, l, matElem = gm.LoadFileThree(fileName+'_pos')
    
    matElem[matElem < 1e-5] = np.NaN
    
    sm.set_array(matElem)
    colors = sm.to_rgba(matElem)
    
    for i in np.arange(0,len(energy)):
        rect = patches.Rectangle((l[i] - 0.45, energy[i]), width=0.9, height=0.003, color=colors[i])
        ax.set_facecolor(cmap(0))
        ax.add_patch(rect)

    ax.set_xlim(-max(l) - 0.5, max(l) + 0.5)
    ax.set_ylim(min(energy) - 0.05, max(energy) + 0.1)
    
    plt.xticks(np.arange(-max(l),max(l)+1))
    
    plt.colorbar(sm)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$\omega_n$')
    
    plt.savefig(fileName+'.pdf')
