import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import GenericModule as gm
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
    
def SpatialCorrelation(site, eigenstate, basisVectors, p):
    """
    Calculates the p-points spatial correlation function <n_{site}(n_{site} - 1)>
    """
    cFunction = 0.
    
    if p==2:
        ithElems = basisVectors[:,site]
        basisIndices = np.nonzero(ithElems)[0]
        
        ni = ithElems[basisIndices]

        cFunction = np.sum( (ni*(ni-1)) * np.abs(eigenstate[basisIndices])**2 )
        
    elif p==3:
        ithElems = basisVectors[:,site]
        basisIndices = np.nonzero(ithElems)[0]
        
        ni = ithElems[basisIndices]

        cFunction = np.sum( (ni*(ni-1)*(ni-2)) * np.abs(eigenstate[basisIndices])**2 )
    
    return cFunction

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
parser.add_argument('-p', type=int, default=2, help='p-points spatial correlation function to calculate (default=2)')
parser.add_argument('--conf', type=float, help='harmonic trap confinement strength (v0) as v0 * r^2')
parser.add_argument('--gamma', type=float, default=2, help='trap steepness (g) as v0 * (r)^g (default=2)')
parser.add_argument('--alpha', type=float, help='magnetic flux density as alpha=p/q')
parser.add_argument('--hardcore', type=int, nargs='?', const=1, default=0, help='hardcore bosons mode')
parser.add_argument('--neigenstate', type=int, help='index of the eigenstate for the average of the correlation function (e.g. 0 is the groundstate)')
parser.add_argument('--corrvsflux', type=int, nargs='?', const=1, default=0, help='calculate the correlation function as function of the magnetix flux alpha (--alphainit and --alphafinal to be given)')
parser.add_argument('--alphainit', type=float, default=0.1, help='starting flux to calculate the correlation function with (default=0.1)')
parser.add_argument('--alphafinal', type=float, default=0.2, help='final flux to calculate the correlation function with (default=0.2)')
parser.add_argument('--alphastep', type=float, default=0.01, help='step to increase alpha (default=0.01)')
# Time dependent correlation function
parser.add_argument('--timedep', type=int, nargs='?', const=1, default=0, help='flag to use the time evolved eigenstate --neigenstate to calculate C(t), e.g. C_2(t)=<psi_t|n(n-1)|psi_t> (to be specified --tmax and --dt)')
parser.add_argument('--tmax', type=float, default=5.0, help='maximum time of the evolved state psi_t to calculate C(t)')
parser.add_argument('--dt', type=float, default=0.01, help='timestep for the evolved states')
# Laser parameters
parser.add_argument('-r0', type=float, help='laser r0 parameter')
parser.add_argument('--angmom', type=int, default=0, help='angular momentum (l) injected by the LG beam (default=0)')
parser.add_argument('--omega', type=float, default=0.0, help='energy injected by the LG beam (default=0)')
parser.add_argument('--epsilon', type=float, default=1.0, help='the intensity of the Laguerre-Gauss perturbation term as epsilon*O(t)')
# Multiple correlation function calculation (adaptive from the maximum overlap eigenstates)
parser.add_argument('--corrmultipleadaptive', type=int, nargs='?', const=1, default=0, help='calculates C_p for a bunch of states retrieved from the maxoverlaps file in StatesOverlap.py (specify all the right parameters necessary to find the _maxoverlaps file, i.e. -U, -U3, --neigenstate etc.)')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U = args.U
if args.U3 is not None: U3 = args.U3
if args.alpha is not None: FluxDensity = args.alpha
if args.r0 is not None: r0 = args.r0
if args.conf is not None: trapConf = args.conf
if args.neigenstate is not None: nEigenstate = args.neigenstate

gamma = args.gamma
p = args.p

angMom = args.angmom
omega = args.omega
eps = args.epsilon

tMax = args.tmax
dt = args.dt

if p>3:
    print('Only p=2, p=3 spatial correlation functions are implemented.')
    sys.exit(1)

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True
    
if args.corrvsflux == 0:
    corrVsFlux = False
elif args.corrvsflux == 1:
    corrVsFlux = True
    alphaInit = args.alphainit
    alphaFinal = args.alphafinal
    alphaStep = args.alphastep

Ns = L*L
c = gm.FindCenter(L)

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

if ((corrVsFlux == False) and (args.timedep == 0) and (args.corrmultipleadaptive == 0)):
    # Load the eigenvectors
    print('Loading the eigenvector n={nEigenstate}...')
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N)
    eigVec = gm.LoadVector(fileName)

    cFunction = np.zeros((Ns))
    cFuncSum = 0.0
    for i in np.arange(0,Ns):
        cFunction[i] = SpatialCorrelation(i, eigVec, basisVectors, p)
        cFuncSum = cFuncSum + cFunction[i]
        
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, corrFunction=p, N=N)
    SaveLocalDensity(fileName, cFunction, L)
    cFuncSum = cFuncSum / N
    print(f'Spatial sum of the correlation function C_{p}/N = {cFuncSum}')
    
if corrVsFlux == True:
    for a in np.arange(alphaInit,alphaFinal,alphaStep):
        print('Loading the eigenvector n={nEigenstate} for flux alpha={a}...')
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=round(a,2), N=N)
        eigVec = gm.LoadVector(fileName)
        
        cFunction = 0.
        for i in np.arange(0,Ns):
            cFunction = cFunction + SpatialCorrelation(i, eigVec, basisVectors, p)
        
        cFunction = cFunction / N
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=alphaInit, corrFunction=p, N=N)
        gm.SaveTwoColFile(fileName, round(a,2), cFunction)
        print(f'Saving {p}-points correlation function for alpha={round(a,2)} on file {fileName}...')
        
if args.timedep == 1:
    nbrOfSteps = int(tMax/dt)
    time = []
    corrFunc = []
    for ti in np.arange(1,nbrOfSteps):
        print(f'Loading the eigenvector n={nEigenstate} evolved at time t={ti*dt}...')
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N, evolvedState=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=eps, timeEvolOmega=omega, timeEvolState=round(ti*dt,2))
        eigVec = gm.LoadVector(fileName)
        
        cFunction = 0.
        for i in np.arange(0,Ns):
            cFunction = cFunction + SpatialCorrelation(i, eigVec, basisVectors, p)

        cFunction = cFunction / N
        
        time.append(ti*dt)
        corrFunc.append(cFunction)
        
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N, r0=r0, timeEvolAngMom=angMom, evolvedState=True, timeEvolEps=eps, timeEvolOmega=omega, timeEvolState=round(ti*dt,2), corrFunction=p)
    gm.SaveArraysTwoColFile(fileName, np.array(time), np.array(corrFunc))
    print(f'Saving {p}-points correlation function for time t={ti*dt} on file {fileName}...')

if args.corrmultipleadaptive == 1:
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N, maxOverlap=True)
    U3vals, maxOverlapIndex, maxOverlaps = gm.LoadFileThree(fileName)
    
    corrFunc = []
    for m in np.arange(0,len(U3vals)):
        fileName = gm.GenFilename(hardcore, L, J, round(8.1-U3vals[m],6), trapConf, gamma, int(maxOverlapIndex[m]), U3=round(U3vals[m],6), alpha=FluxDensity, N=N)
        eigVec = gm.LoadVector(fileName)
        
        print(f'U={round(8.1-U3vals[m],6)},U3={round(U3vals[m],6)}: calculating C_{p} for the n={int(maxOverlapIndex[m])} state...')
        
        cFunction = 0.
        for i in np.arange(0,Ns):
            cFunction = cFunction + SpatialCorrelation(i, eigVec, basisVectors, p)
            
        cFunction = cFunction / N
        corrFunc.append(cFunction)

    print(f'Correlation functions C_{p}:')
    print(corrFunc)
    
    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N, corrFunction=p) + '_adaptive'
    gm.SaveArraysTwoColFile(fileName, U3vals, corrFunc)
    print(f'Correlation functions saved in ---> {fileName}')
