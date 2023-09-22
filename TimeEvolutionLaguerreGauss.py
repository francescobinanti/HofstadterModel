import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import GenericModule as gm
import HofstadterThreeBody as HH

def LoadSpectrum(fileName):
    """
    Load the energy spectrum
    """
    fileName = fileName + '.dat'
    return np.loadtxt(fileName)
    
def SaveAbsorptionSpectrum(fileName, deltaEnergies, l, matElements):
    """
    Save the absorption spectrum {energy -- angular momentum -- matrix element} on a file
    """
    fileName = fileName + '.dat'
    
    lSeq = [l for i in np.arange(0,len(deltaEnergies))]
    dataStack = np.column_stack((deltaEnergies, lSeq, matElements))
    
    print(f'Saving the absorption spectrum for l={l} on file "{fileName}"...')
    with open(fileName,"ab") as fileDesc:
        np.savetxt(fileDesc, dataStack)
        
def LaguerrePolynomial(n, l, x):
    """
    Generates the Laguerre polynomial of radial order n and fixed angular order l in the position x: L_nˆl(x)
    """
    laguerrePol = 0
    for i in np.arange(0,n+1):
        laguerrePol = laguerrePol + ((-1.)**i) * sp.special.comb(n+l,n-i) * ((x**i)/math.factorial(i))
    
    return laguerrePol

def GenerateLaguerreGaussTerms(L, r0, l, eps, omega, tMax, dt, n=0):
    """
    It creates the Laguerre-Gauss terms for the Hamiltonian H(t) at time t
    """
    print(f'Generating LG modes at all times')
    c = HH.FindCenter(L)
    totalSteps = int(tMax / dt)
    # These are the the matrix elements O(t) for all times on the lattice LGCoeff[time][space]
    LGCoeff = np.zeros((totalSteps, L*L), dtype=complex)
    OAMPhases = np.zeros((L*L), dtype=complex)
    LGPreFactor = np.sqrt(2*math.factorial(n)/( np.pi * np.math.factorial(n + np.abs(l)) ))
    #print(f'PreFactorLG={LGPreFactor}')
    tmpNFactor = 0
    NFactor = 0

    for ti in np.arange(1,totalSteps):
        t = ti * dt
        timePhase = gm.Phase(-omega * t)
        for y in np.arange(0,L):
            for x in np.arange(0,L):
                siteIndex = x + L*y
                radius = gm.Radius(x,y,c)
                ang = np.arctan2(y-c,x-c)
                OAMPhases[siteIndex] = gm.Phase(ang * l)
                #print(f'Phase(i={siteIndex}) = {OAMPhases[siteIndex]}')
                
                if (n == 0):
                    laguerrePolynomial = 1.
                else:
                    laguerrePolynomial = LaguerrePolynomial(n, l, radius)
                    
                LGCoeff[ti,siteIndex] = LGPreFactor * ( (radius/r0)**np.abs(l) ) * np.exp( -(radius*radius) / (2*r0*r0) ) * laguerrePolynomial
                #print(f'f(i={siteIndex})={(eps * LGCoeff[ti,siteIndex] * OAMPhases[siteIndex])}')
                tmpNFactor = LGCoeff[ti,siteIndex]
                #print(tmpNFactor)
                if (tmpNFactor > NFactor):
                    NFactor = tmpNFactor
                    
        #print(f'NFactor={NFactor}')
        #print(OAMPhases)
        LGCoeff[ti,:] = eps * (LGCoeff[ti,:] / NFactor) * ( (OAMPhases * timePhase) + np.conj(OAMPhases * timePhase) )
        #print(LGCoeff)
        
    #print(f'Max factor = {NFactor}')
            
    return LGCoeff
    
def BuildTimeHamiltonian(H_0, LGTerms, binBasis, t):
    """
    It creates the Hamiltonian H(t) = H_0 + O(t) with the time dependent LG perturbation
    """
    H_t = H_0.copy()
    #print(f'Shape of Ht = {np.shape(H_t)}')
    
    for state in binBasis:
        stateInt = HH.PomeranovTag([state])
        stateString = HH.BitArrayToString(state)
        
        filledSites = list(i for i, x in enumerate(stateString) if x != '0')
        for idx in filledSites:
            H_t[stateInt-1,stateInt-1] += state[idx] * LGTerms[t,idx]
            
    return H_t
    

def BuildAllTimesHamiltonian(H_0, LGTerms, binBasis, intBasis, tMax, dt, hardcore=True):

    timeSteps = int(tMax / dt)
    HArray = [H_0.copy() for i in np.arange(0,tMax,dt)]
    intArray = {subArray: i for i, subArray in enumerate(intBasis)}
    
    for state in binBasis:
        if hardcore == False:
            stateInt = HH.PomeranovTag([state]) - 1
        else:
            stateTag = HH.AssignTagToState(state,n=1)
            stateInt = intArray[stateTag]
            
        stateString = HH.BitArrayToString(state)
        filledSites = list(i for i, x in enumerate(stateString) if x != '0')
        for idx in filledSites:
            for ti in np.arange(1,timeSteps):
                HArray[ti][stateInt,stateInt] += state[idx] * LGTerms[ti,idx]
                
    return HArray
    
    
"""
def BuildAllTimesHamiltonian(H_0, LGTerms, binBasis, intBasis, tMax, dt, hardcore=True):

    timeSteps = int(tMax / dt)
    HArray = [H_0.copy() for i in np.arange(0,tMax,dt)]
    intArray = {subArray: i for i, subArray in enumerate(intBasis)}
    positionMatElem = []
    filledSites = []
    
    start = time.time()
    for state in binBasis:
        if hardcore == False:
            stateInt = HH.PomeranovTag([state]) - 1
            positionMatElem.append(stateInt)
        else:
            stateTag = HH.AssignTagToState(state,n=1)
            stateInt = intArray[stateTag]
            positionMatElem.append(stateInt)
            
        stateString = HH.BitArrayToString(state)
        filledSites.append(list(i for i, x in enumerate(stateString) if x != '0'))
    gm.TimePrint(start)
        
    
    print('Positions:')
    print(positionMatElem[0])
    print('====================')
    print('Filled sites:')
    print(filledSites[0])
    print('LG modes given by filled sites:')
    #print([np.sum(binBasis[positionMatElem[n]][filledSites[n]] * LGTerms[4,filledSites[n]]) for n in np.arange(0,len(positionMatElem))])
    
    for n in np.arange(0,len(positionMatElem)):
        print(np.sum(np.sum(binBasis[positionMatElem[n]][filledSites] * LGTerms[1:timeSteps,filledSites], axis=1),axis=1))
        HArray[:][positionMatElem[n],positionMatElem[n]] = np.sum(np.sum(binBasis[positionMatElem[n]][filledSites] * LGTerms[1:timeSteps,filledSites], axis=1),axis=1)
            
    return HArray
    """
    
def TimeEvolution(H_t, state, timeStep):
    """
    Perform the time evolution of a state with the Hamiltonian H(t) (t fixed)
    by mean of expm() function, i.e. e^{-i H(t) dt}|Psi(t-dt)> = 1 - i H(t) dt + 0.5 * (H(t) * dt)^2 + ...
    where the order is set by the expm() function itself (?)
    """
    evolvedState = sp.sparse.linalg.expm_multiply(-complex(0.,1.) * H_t * timeStep, state)
    #evolvedNorm = sp.linalg.norm(evolvedState)
    #print(f'Norm of state after time evolution |psi|ˆ2 = {evolvedNorm}')
    return evolvedState
    
def CalcExcitationFraction(evolvedState, groundState, t):
    """
    Calculate the excitation fraction out of a time evolution like 1 - |<psi_t|psi_0>|ˆ2
    """
    excFraction = 1. - np.abs(np.vdot(evolvedState,groundState))**2.
    
    return excFraction
    
def GenSquareSites(L, s):
    """
    Given a certain integer s, it generates all the lattice coordinates belonging to the ring squares out of the s-th one and within
    that region, separately.
    For instance with s=2 in a 6x6 lattice we would have the sites marked with the "o" and "x" separately:
                    o  o  o  o  o  o
                    o  x  x  x  x  o
                    o  x  x  x  x  o
                    o  x  x  x  x  o
                    o  x  x  x  x  o
                    o  o  o  o  o  o
    """
    config = np.zeros((L, L), dtype=int)
    lattice = np.zeros((L, L), dtype=int)

    # Define the starting point and the direction of the spiral
    x, y = 0, 0
    dx, dy = 0, 1

    # Fill the table with the spiral sequence
    for i in range(1, L*L + 1):
        config[x, y] = i
        lattice[(i-1)%L, (i-1)//L] = i
        nx, ny = x + dx, y + dy
        if 0 <= nx < L and 0 <= ny < L and config[nx, ny] == 0:
            x, y = nx, ny
        else:
            dx, dy = dy, -dx
            x, y = x + dx, y + dy
            
    config = -(config - (L*L))
    lattice = np.rot90(lattice - 1)
    
    #print(config)
    #print(lattice)
    
    if L%2 == 0:
        boolTable = (config >= (2*s)**2)
        #print(config >= (2*s)**2)
    else:
        boolTable = (config >= ((2*s-1)**2))
        #print(config >= ((2*s-1)**2))
        
    finalCoord = lattice[boolTable]
    finalCoordNeg = lattice[~boolTable]
    #print(finalCoord)
    #print(finalCoordNeg)
    
    return finalCoord, finalCoordNeg

    
def CalcDensityTime(site, evolvedState, basisVectors):
    """
    It calculates the density <psi_0(t)|n_{site}|psi_0(t)> on site=site for the ground state evolved in time.
    """
    # Select the ith elements to be checked in the basis vectors
    ithElems = basisVectors[:,site]
    basisIndices = np.nonzero(ithElems)[0]
    
    #print(basisVectors)
    #print(basisIndices)
    
    totalDensity = np.sum(np.abs(evolvedState[basisIndices])**2)
    
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
parser.add_argument('-r0', type=float, help='Laguerre-Gauss radius parameter (for a pure gaussian beam it is the gaussian dispersion)')
parser.add_argument('-n', type=int, default=0, help='radial order of the Laguerre polynomials used in the spatial mode of the LG beam')
parser.add_argument('--angmom', type=int, default=0, help='angular momentum (l) injected by the LG beam (default=0)')
parser.add_argument('--omega', type=float, default=0.0, help='energy injected by the LG beam (default=0)')
parser.add_argument('--epsilon', type=float, default=1.0, help='the intensity of the Laguerre-Gauss perturbation term as epsilon*O(t)')
parser.add_argument('--tmax', type=float, default=1.0, help='time at which the time evolution has to stop (default=1.0)')
parser.add_argument('--dt', type=float, default=0.01, help='timestep (default=0.01)')
# Optional
parser.add_argument('--density', type=int, nargs='?', const=1, default=0, help='calculate the variation of the ground state local density in time (at the edge), i.e. rho(t) - rho(0)')
parser.add_argument('--squares', type=int, default=2, help='specify which square ring of the lattice (starting with 1 for the inner center) has to be considered as the beginning of the "edge" for the calculation of the density variation (default=2, namely we consider from s=3 onwards)')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U3 = args.U
if args.U3 is not None: U3 = args.U3
if args.r0 is not None: r0 = args.r0
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf

gamma = args.gamma
angMom = args.angmom
omega = args.omega
eps = args.epsilon
tMax = args.tmax
dt = args.dt

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True
    
if args.density == 0:
    densityFlag = False
elif args.density == 1:
    densityFlag = True
    
s = args.squares

# Laser parameters
nLG = args.n
print(f'r_0 = {r0}')
print(f'LG radial order n={nLG}')

Ns = L*L
c = HH.FindCenter(L)

print('Generating basis vectors...')
if (hardcore == True):
    # Generate the hardcore basis
    print('Hard-core bosons')
    basisStrings = [string for string in HH.GenerateHardcoreBasis(N, Ns)]
    basisVectors = np.array([ [int(bit) for bit in string] for string in basisStrings ])
    intBasisVectors = HH.AssignTagToState(basisVectors, 0)
    Dim = len(intBasisVectors)
else:
    print(f'Soft-core bosons, U={U}, U3={U3}')
    # Generate the softcore basis
    basisStrings = [string for string in HH.GenerateBasis(N, Ns)]
    #print(basisStrings)
    basisVectors = np.array([ [int(bit) for bit in string] for string in basisStrings ])
    intBasisVectors = HH.PomeranovTag(basisVectors)
    #print(intBasisVectors)
    Dim = len(intBasisVectors)

print(f'Hilbert space dimension={Dim}')

# Load the Hamiltonian at time t=0
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, U3=U3, alpha=FluxDensity, N=N)
H_0 = gm.LoadMatrix(fileName)

# Load the ground state
print('Loading the ground state vector...')
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N)
groundStateVec = gm.LoadVector(fileName)

LGTerms = GenerateLaguerreGaussTerms(L, r0, angMom, eps, omega, tMax, dt, n=nLG)
#print(f'LGTerms shape = {np.shape(LGRadial)}')
#print(LGTerms)

print('-------- Laser parameters --------')
print(f'omega={omega}')
print(f'angular momentum={angMom}')
print(f'epsilon={eps}')
print('----------------------------------')

print(f'Building the H(t) for all times within t=[0,{tMax}]...')
start = time.time()
H = BuildAllTimesHamiltonian(H_0, LGTerms, basisVectors, intBasisVectors, tMax, dt, hardcore)
gm.TimePrint(start)

"""
# CHECK: diagonalize H(t) at different times and see if the GS is periodic with omega of the laser
for ti in np.arange(0, int(tMax / dt)):
    E, eVec = HH.diagH(H[ti], 1)
    print(f'Ground state of H[{ti}]:')
    print(E)
"""

# Time evolution
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, excFrac=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=eps, timeEvolOmega=omega)
print('Starting time evolution...')
evolvedState = groundStateVec.copy()

if densityFlag == True:
    evolvedEdgeDensity = 0
    evolvedBulkDensity = 0
    groundStateEdgeDensity = 0
    groundStateBulkDensity = 0

    edgeSiteSquares, bulkSiteSquares = GenSquareSites(L, s)

    for site in edgeSiteSquares:
        groundStateEdgeDensity += CalcDensityTime(site, groundStateVec, basisVectors)
    for site in bulkSiteSquares:
        groundStateBulkDensity += CalcDensityTime(site, groundStateVec, basisVectors)

for ti in np.arange(1, int(tMax / dt)):
    evolvedState = TimeEvolution(H[ti].tocsc(), evolvedState, dt)
    excFraction = CalcExcitationFraction(evolvedState, groundStateVec, ti*dt)
    gm.SaveTwoColFile(fileName, ti*dt, excFraction)
    
    if densityFlag == True:
        for site in edgeSiteSquares:
            evolvedEdgeDensity += CalcDensityTime(site, evolvedState, basisVectors)
        for site in bulkSiteSquares:
            evolvedBulkDensity += CalcDensityTime(site, evolvedState, basisVectors)
            
        evolvedEdgeDensity = evolvedEdgeDensity - groundStateEdgeDensity
        evolvedBulkDensity = evolvedBulkDensity - groundStateBulkDensity
        
        fileNameDensity = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, densityEvolution=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=eps, timeEvolOmega=omega)
        gm.SaveTwoColFile(fileNameDensity+'_edge', ti*dt, evolvedEdgeDensity)
        gm.SaveTwoColFile(fileNameDensity+'_bulk', ti*dt, evolvedBulkDensity)
        
        evolvedEdgeDensity = 0
        evolvedBulkDensity = 0
    
