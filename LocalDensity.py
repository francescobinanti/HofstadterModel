import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import HofstadterThreeBody as HHModule
    
def SaveLocalDensity(fileName, density, L):
    """
    Save the absorption spectrum {energy -- angular momentum -- matrix element} on a file
    """
    fileName = fileName + '.dat'
    
    sites = [i for i in np.arange(0,len(density))]
    x = [ site%L for site in sites ]
    y = [ site//L for site in sites ]
    dataStack = np.column_stack((x, y, density))
    
    print(f'Saving the local density in file "{fileName}"...')
    with open(fileName,"ab") as fileDesc:
        np.savetxt(fileDesc, dataStack)
    
def CalcLocalDensity(site, eigenstate, basisVectors):
    """
    It calculates the density <psi_n|n_{site}|psi_n> on site=site for the state n=nEigen.
    """
    # Select the ith elements to be checked in the basis vectors
    ithElems = basisVectors[:,site]
    basisIndices = np.nonzero(ithElems)[0]
    
    #print(basisVectors)
    #print(basisIndices)
    
    totalDensity = np.sum(np.abs(eigenstate[basisIndices])**2)
    
    return totalDensity


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
parser.add_argument('--conf', type=float, help='harmonic trap confinement strength (v0) as v0 * r^2')
parser.add_argument('--gamma', type=float, default=2, help='trap steepness (g) as v0 * (r)^g (default=2)')
parser.add_argument('--alpha', type=float, help='magnetic flux density as alpha=p/q')
parser.add_argument('--hardcore', type=int, nargs='?', const=1, default=0, help='hardcore bosons mode')
parser.add_argument('--neigenstate', type=int, help='index of the eigenstate for which the local density has to be calculated (e.g. 0 is the groundstate)')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U3 = args.U
if args.U3 is not None: U3 = args.U3
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf
if args.neigenstate is not None: nEigenstate = args.neigenstate

gamma = args.gamma

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True

Ns = L*L
c = HHModule.FindCenter(L)

print('Generating basis vectors...')
if (hardcore == True):
    # Generate the hardcore basis
    print('Hard-core bosons')
    basisStrings = [string for string in HHModule.GenerateHardcoreBasis(N, Ns)]
    basisVectors = np.array([ [int(bit) for bit in string] for string in basisStrings ])
    intBasisVectors = HHModule.AssignTagToState(basisVectors, 0)
    Dim = len(intBasisVectors)
else:
    print(f'Soft-core bosons, U={U}, U3={U3}')
    # Generate the softcore basis
    basisStrings = [string for string in HHModule.GenerateBasis(N, Ns)]
    #print(basisStrings)
    basisVectors = np.array([ [int(bit) for bit in string] for string in basisStrings ])
    intBasisVectors = HHModule.PomeranovTag(basisVectors)
    #print(intBasisVectors)
    Dim = len(intBasisVectors)

print(f'Hilbert space dimension={Dim}')

# A tuple of (idx for states having non zero elements, idx position of where the non-zero elements are).
#This is needed for the creation of the O_l operator (namely the sum_i n_i operator matrix)
print('Creating the index vectors for the LG operator...')
statesWithNonZeroSites = np.nonzero(basisVectors)
#print(statesWithNonZeroSites)

# Load the eigenvectors
print('Loading the eigenvector n={nEigenstate}...')
fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N)
eigVec = HHModule.LoadVector(fileName)

density = np.zeros((Ns))
for i in np.arange(0,Ns):
    density[i] = CalcLocalDensity(i, eigVec, basisVectors)
    
fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, localDensity=True, N=N)
SaveLocalDensity(fileName, density, L)

