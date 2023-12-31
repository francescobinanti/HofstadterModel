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
    c = HH.FindCenter(L)
    totalSteps = int(tMax / dt)
    # These are the the matrix elements O(t) for all times on the lattice LGCoeff[time][space]
    LGCoeffZeroTime = np.zeros((L*L), dtype=complex)
    LGCoeff = np.zeros((totalSteps, L*L), dtype=complex)
    OAMPhases = np.zeros((L*L), dtype=complex)
    LGPreFactor = np.sqrt(2*math.factorial(n)/( np.pi * np.math.factorial(n + np.abs(l)) ))
    #print(f'PreFactorLG={LGPreFactor}')
    tmpNFactor = 0
    NFactor = 0

    if tMax != 0.0:
        print(f'Generating LG modes at all times...')
        for ti in np.arange(0,totalSteps):
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
            
            return LGCoeff
    else:
        print(f'Generating LG modes at time t=0...')
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
                    
                LGCoeffZeroTime[siteIndex] = LGPreFactor * ( (radius/r0)**np.abs(l) ) * np.exp( -(radius*radius) / (2*r0*r0) ) * laguerrePolynomial
                #print(f'f(i={siteIndex})={(eps * LGCoeff[ti,siteIndex] * OAMPhases[siteIndex])}')
                tmpNFactor = LGCoeffZeroTime[siteIndex]
                #print(tmpNFactor)
                if (tmpNFactor > NFactor):
                    NFactor = tmpNFactor
                    
        #print(f'NFactor={NFactor}')
        #print(OAMPhases)
        LGCoeffZeroTime[:] = eps * (LGCoeffZeroTime[:] / NFactor) * ( (OAMPhases) + np.conj(OAMPhases) )
        
        return LGCoeffZeroTime
        
def GenerateLaguerreGaussTermsIncreaseEpsilon(L, r0, l, epsilonInit, epsilonStep, omega, tMax, dt, n=0):
    """
    Same as GenerateLaguerreGaussTerms() but increasing epsilon at each timestep
    """
    c = HH.FindCenter(L)
    totalSteps = int(tMax / dt)
    # These are the the matrix elements O(t) for all times on the lattice LGCoeff[time][space]
    LGCoeffZeroTime = np.zeros((L*L), dtype=complex)
    LGCoeff = np.zeros((totalSteps, L*L), dtype=complex)
    OAMPhases = np.zeros((L*L), dtype=complex)
    LGPreFactor = np.sqrt(2*math.factorial(n)/( np.pi * np.math.factorial(n + np.abs(l)) ))
    #print(f'PreFactorLG={LGPreFactor}')
    tmpNFactor = 0
    NFactor = 0
    eps = epsilonInit

    print(f'Generating LG modes at all times increasing epsilon by {epsilonStep} at each timestep...')
    for ti in np.arange(0,totalSteps):
        t = ti * dt
        timePhase = gm.Phase(-omega * t)
        eps = eps + epsilonStep
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
    
def BuildZeroTimeHamiltonian(H_0, LGTerms, binBasis, intBasis, hardcore=True):
    """
    Builds a Hamiltonian H = H_0 + O_l where the laser frequency omega=0 (so time-independent)
    """
    HPerturbed = H_0.copy()
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
            HPerturbed[stateInt,stateInt] += state[idx] * LGTerms[idx]
                
    return HPerturbed
    
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
parser.add_argument('--epsilonspectrum', type=int, nargs='?', const=1, default=0, help='calculate the energy spectrum as a function of epsilon (laser power)')
parser.add_argument('--epsilonsavestate', type=float, help='the groundstate of H = H_0 + eps*O_l to be saved during the --epsilonspectrum operation')
parser.add_argument('--epsiloninit', type=float, default=0.1, help='initial value of epsilon for the --epsilonspectrum option (default=0.1)')
parser.add_argument('--epsilonfinal', type=float, default=1.0, help='final value of epsilon for the --epsilonspectrum option (default=1.0)')
parser.add_argument('--epsilonstep', type=float, default=0.1, help='step for the increase of epsilon for the --epsilonspectrum option (default=0.1)')
parser.add_argument('--nbreigenstates', type=int, default=10, help='number of states to calculate when using the --epsilonspectrum or --epsilonevolutiontime option (default=10)')
parser.add_argument('--savestates', type=int, nargs='?', const=1, default=0, help='save ALL the eigenstates calculated in the --epsilonspectrum option')
parser.add_argument('--tmax', type=float, default=1.0, help='time at which the time evolution has to stop (default=1.0)')
parser.add_argument('--dt', type=float, default=0.01, help='timestep (default=0.01)')
parser.add_argument('--savetimestates', type=float, default=0.0, help='specify the timesteps at which the time evolved wavefunction should be saved in a file')
parser.add_argument('--excfraction', type=int, nargs='?', const=1, default=0, help='calculate the excitation fraction 1 - |<psi_0|psi_t>|ˆ2')
parser.add_argument('--epsilonevolutiontime', type=int, nargs='?', const=1, default=0, help='flag to run the H_0 + epsilon(t)*O_l(t) evolution (epsilon increases at each timestep starting from zero). Need to specify the laser parameters and the time evolution parameters --tmax, --dt.')
# Optional
parser.add_argument('--density', type=int, nargs='?', const=1, default=0, help='calculate the variation of the ground state local density in time (at the edge), i.e. rho(t) - rho(0)')
parser.add_argument('--squares', type=int, default=2, help='specify which square ring of the lattice (starting with 1 for the inner center) has to be considered as the beginning of the "edge" for the calculation of the density variation (default=2, namely we consider from s=3 onwards)')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U = args.U
if args.U3 is not None: U3 = args.U3
if args.r0 is not None: r0 = args.r0
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf
if args.epsilonsavestate is not None: epsSave = args.epsilonsavestate

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
    
if args.epsilonspectrum == 1:
    nbrEigenstates = args.nbreigenstates
    
epsilonInit = args.epsiloninit
epsilonFinal = args.epsilonfinal
epsilonStep = args.epsilonstep
    
s = args.squares

# Laser parameters
nLG = args.n
print(f'r_0 = {r0}')
print(f'LG radial order n={nLG}')

Ns = L*L
c = HH.FindCenter(L)

# Load the Hamiltonian at time t=0
print('Loading the Hofstadter Hamiltonian...')
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, U3=U3, alpha=FluxDensity, N=N)
H_0 = gm.LoadMatrix(fileName)

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

# Load the ground state
print('Loading the ground state vector...')
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N)
groundStateVec = gm.LoadVector(fileName)

#print(f'LGTerms shape = {np.shape(LGRadial)}')
#print(LGTerms)

print('-------- Laser parameters --------')
print(f'omega={omega}')
print(f'angular momentum={angMom}')
print(f'epsilon={eps}')
print('----------------------------------')

"""
# CHECK: diagonalize H(t) at different times and see if the GS is periodic with omega of the laser
for ti in np.arange(0, int(tMax / dt)):
    E, eVec = HH.diagH(H[ti], 1)
    print(f'Ground state of H[{ti}]:')
    print(E)
"""

# Time evolution
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
        
if args.epsilonspectrum != 0:
    tMax = 0.0
    print('---- Spectrum vs epsilon ----')
    for epsilon in np.arange(epsilonInit,epsilonFinal,epsilonStep):
        print(f'epsilon={epsilon}')
        LGTerms = GenerateLaguerreGaussTerms(L, r0, angMom, epsilon, omega, tMax, dt, n=nLG)
        H = BuildZeroTimeHamiltonian(H_0, LGTerms, basisVectors, intBasisVectors, hardcore)
    
        eVals, eVecs = HH.diagH(H, nbrEigenstates)
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, spectrum=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=round(epsilon,2), timeEvolOmega=omega)
        gm.SaveSpectrum(fileName, eVals)
        
        if args.savestates == 1:
            for nEigenstate in np.arange(0, nbrEigenstates):
                fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N, r0=r0, timeEvolAngMom=angMom, timeEvolEps=round(epsilon,2), timeEvolOmega=omega)
                gm.SaveVector(fileName, eVecs[:,nEigenstate])
        
        if args.epsilonsavestate is not None:
            if round(epsilon,2) == epsSave:
                fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, r0=r0, timeEvolAngMom=angMom, timeEvolEps=round(epsilon,2), timeEvolOmega=omega)
                gm.SaveVector(fileName, eVecs[0])

if (args.epsilonspectrum == 0) and (args.epsilonevolutiontime == 0):
    if omega != 0.0:
        LGTerms = GenerateLaguerreGaussTerms(L, r0, angMom, eps, omega, tMax, dt, n=nLG)
        print(f'Building the H(t) for all times within t=[0,{tMax}]...')
        start = time.time()
        H = BuildAllTimesHamiltonian(H_0, LGTerms, basisVectors, intBasisVectors, tMax, dt, hardcore)
        gm.TimePrint(start)
    else:
        LGTerms = GenerateLaguerreGaussTerms(L, r0, angMom, eps, omega, 0.0, dt, n=nLG)
        print(f'Building the perturbed Hamiltonian H = H_0 + O_l...')
        start = time.time()
        H = BuildZeroTimeHamiltonian(H_0, LGTerms, basisVectors, intBasisVectors, hardcore)
        gm.TimePrint(start)
    
    print('Starting time evolution...')
    
    for ti in np.arange(1, int(tMax / dt)):
        if omega != 0.0:
            evolvedState = TimeEvolution(H[ti].tocsc(), evolvedState, dt)
        else:
            evolvedState = TimeEvolution(H.tocsc(), evolvedState, dt)
            
        if args.excfraction == 1:
            fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, excFrac=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=eps, timeEvolOmega=omega)
            excFraction = CalcExcitationFraction(evolvedState, groundStateVec, ti*dt)
            gm.SaveTwoColFile(fileName, ti*dt, excFraction)
        
        if args.savetimestates != 0.0:
            stepSave = int(tMax/args.savetimestates)
            if ti%stepSave == 0:
                fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, evolvedState=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=eps, timeEvolOmega=omega, timeEvolState=ti*dt)
                gm.SaveVector(fileName, evolvedState)
        
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
            
if args.epsilonevolutiontime == 1:
    print('---- Time evolution with increasing laser power at each timestep ----')
    LGTerms = GenerateLaguerreGaussTermsIncreaseEpsilon(L, r0, angMom, epsilonInit, epsilonStep, omega, tMax, dt, n=nLG)
    print(f'Building the H(t) for all times within t=[0,{tMax}]...')
    start = time.time()
    H = BuildAllTimesHamiltonian(H_0, LGTerms, basisVectors, intBasisVectors, tMax, dt, hardcore)
    gm.TimePrint(start)
    eps = epsilonInit
    for ti in np.arange(0, int(tMax / dt)):
        eps = eps + epsilonStep
    
        eVals, eVecs = HH.diagH(H[ti], nbrEigenstates)
        fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, U3=U3, alpha=FluxDensity, N=N, spectrum=True, r0=r0, timeEvolAngMom=angMom, timeEvolEps=round(eps,2), timeEvolOmega=omega, timeEvolState=round(ti*dt,2))
        gm.SaveSpectrum(fileName, eVals)
