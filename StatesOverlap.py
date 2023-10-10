import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import HofstadterThreeBody as HHModule
import GenericModule as gm
    
def SingleOverlap(firstState, secondState):
    """
    Optimize version of CalculateMatrixElements() using CreateLGMatrixOptimized()
    """
   
    overlap = ( np.vdot(firstState,secondState) )
        
    #print(deltaEnergies)

    return overlap
    
def MultipleOverlap(firstState, eigenstates, nbrSecondStates, threshold=0.2):
    """
    """
    overlaps = []
    largeOverlaps = []
    for n in np.arange(0, nbrSecondStates):
        currOverlap = SingleOverlap(firstState, eigenstates[:,n])
        overlaps.append(currOverlap)
        if np.abs(currOverlap) > threshold:
            largeOverlaps.append(np.abs(currOverlap))
        
    return np.abs(overlaps), largeOverlaps
    

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
parser.add_argument('-Utwo', type=float, help='two-body onsite interaction (only in softcore mode) for the second state')
parser.add_argument('-U3', type=float, help='three-body onsite interaction (only in softcore mode)')
parser.add_argument('-U3two', type=float, help='three-body onsite interaction (only in softcore mode) for the second state')
parser.add_argument('--conf', type=float, help='harmonic trap confinement strength (v0) as v0 * r^2')
parser.add_argument('--gamma', type=float, default=2, help='trap steepness (g) as v0 * (r)^g (default=2)')
parser.add_argument('--alpha', type=float, help='magnetic flux density as alpha=p/q')
parser.add_argument('--alphatwo', type=float, help='magnetic flux density as alpha=p/q for the second state')
parser.add_argument('--hardcore', type=int, nargs='?', const=1, default=0, help='hardcore bosons mode')
parser.add_argument('-n1', type=int, default=0, help='eigenstate number of the first state (e.g. n=0 is the ground state)')
parser.add_argument('-n2', type=int, default=0, help='eigenstate number of the second state (e.g. n=0 is the ground state)')
# Multiple overlap
parser.add_argument('--multipleoverlap', type=int, nargs='?', const=1, default=0, help='to calculate matrix elements with quenched states from H = H_0 + eps*O_l (laser parameters to be specified)')
parser.add_argument('--nbrsecondstates', type=int, help='the number of excited states to do the overlap with <n1|...>')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J

if args.U is not None: U = args.U

if args.Utwo is not None:
    Utwo = args.Utwo
else:
    Utwo = 0

if args.U3 is not None: U3 = args.U3

if args.U3two is not None:
    U3two = args.U3two
else:
    U3two = 0

if args.alpha is not None: FluxDensity = args.alpha

if args.alphatwo is not None:
    FluxDensityTwo = args.alphatwo
else:
    FluxDensityTwo = FluxDensity

if args.conf is not None: trapConf = args.conf
if args.nbrsecondstates is not None: nbrSecondStates = args.nbrsecondstates

if args.n1 is not None: n1 = args.n1
if args.n2 is not None: n2 = args.n2

gamma = args.gamma

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True

Ns = L*L
c = HHModule.FindCenter(L)

Dim = int(gm.HilbertDim(N,Ns,hardcore))

fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, n1, U3=U3, alpha=FluxDensity, N=N)
print(f'Loading the first state --> {fileName}')
firstState = gm.LoadVector(fileName)

if args.multipleoverlap == 1:
    # Load the eigenvectors for the multiple overlaps
    
    print('Loading all the eigenstates for the overlap...')
    eigVec = np.zeros((Dim,nbrSecondStates), dtype=complex)
    for n in np.arange(0,nbrSecondStates):
        fileName = gm.GenFilename(hardcore, L, J, Utwo, trapConf, gamma, n, U3=U3two, alpha=FluxDensityTwo, N=N)
        print(f'--> {fileName}')
        eigVec[:,n] = gm.LoadVector(fileName)

    print('Doing the overlaps...')
    overlaps, largeOverlaps = MultipleOverlap(firstState, eigVec, nbrSecondStates)
    print('Overlaps > 0.2: ')
    print(overlaps)
    

    

