import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import HofstadterThreeBody as HHModule

def LoadSpectrum(fileName):
    """
    Load the energy spectrum
    """
    fileName = fileName + '.dat'
    return np.loadtxt(fileName)

def LoadHamiltonian(fileName):
    """
    Load the Hamiltonian sparse matrix from a file .npz
    """
    fileName = fileName + '.npz'
    return sp.sparse.load_npz(fileName)
    
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

def GenerateLaguerreGauss(L, r0, l, n=0):
    """
    It creates the Laguerre-Gauss spatial mode on the lattice
    """
    print(f'Generating LG modes for angular momentum l={l}')
    c = HHModule.FindCenter(L)
    LGCoeff = np.zeros((L*L), dtype=complex)
    OAMPhases = np.zeros((L*L), dtype=complex)
    LGPreFactor = np.sqrt(2*math.factorial(n)/( np.pi * np.math.factorial(n + np.abs(l)) ))
    #print(f'PreFactorLG={LGPreFactor}')
    tmpNFactor = 0
    NFactor = 0

    for x in np.arange(0,L):
        for y in np.arange(0,L):
            siteIndex = x + L*y
            radius = HHModule.Radius(x,y,c)
            ang = np.arctan2(y-c,x-c)
            OAMPhases[siteIndex] = HHModule.Phase(ang * l)
            
            if (n == 0):
                laguerrePolynomial = 1.
            else:
                laguerrePolynomial = LaguerrePolynomial(n, l, radius)
                
            LGCoeff[siteIndex] = LGPreFactor * ( ((np.sqrt(2)*radius)/r0)**np.abs(l) ) * np.exp( -((radius / r0)**2) ) * laguerrePolynomial
            tmpNFactor = LGCoeff[siteIndex]
            #print(tmpNFactor)
            if (tmpNFactor > NFactor):
                NFactor = tmpNFactor
                
    #print(f'NFactor={NFactor}')
    #print(OAMPhases)
    LGCoeff = (LGCoeff * OAMPhases) / NFactor
    #print(LGCoeff)
            
    return LGCoeff
    
def CreateLGMatrix(site, LGCoeff, basisStrings):
    """
    Create the number operator n_{site} matrix in the many-body basis and stick the
    spatial modes in the
    """
    Dim = len(basisStrings)
    nMatrix = sp.sparse.csr_matrix((Dim,Dim), dtype=complex)
    n = 0
    for state in basisStrings:
        #print(f'creating the LG matrix operator for state={state}')
        if (state[site] != '0'):
            nMatrix[n,n] = int(state[site]) * LGCoeff[site] # this state[site] is for soft-core
        n = n + 1
    
    return nMatrix
    
def CreateLGMatrixOptimized(LGCoeff, basisStrings, nonZeroIndices):
    """
    Optimized version of CreateLGMatrix()
    """
    Dim = len(basisStrings)
    #nMatrix = sp.sparse.csr_matrix((Dim,Dim), dtype=complex)
    
    # Search for the indices
    #startTime = time.time()
    #statesWithNonZeroSite = np.nonzero(basisStrings) # A tuple of (idx for states having non zero elements, idx position of where the non-zero elements are)
    stateIdx = nonZeroIndices[0] # this is the state
    siteIdx = nonZeroIndices[1] # this is the spatial index
    #endTime = time.time()
    
    #print(f'time needed for np.nonzero={(endTime-startTime)/60.}')
    #nMatrix[stateIdx,stateIdx] += basisStrings[stateIdx,siteIdx] #* LGCoeff[siteIdx] # this int(state[site]) is for soft-core
    
    nMatrix = sp.sparse.coo_matrix((basisStrings[stateIdx, siteIdx]*LGCoeff[siteIdx], (stateIdx, stateIdx)), shape=(Dim,Dim), dtype=complex)
    nMatrix = nMatrix.tocsr()
    
    """
    for j in np.arange(0,len(stateIdx)):
        nMatrix[stateIdx[j],stateIdx[j]] += basisStrings[stateIdx[j],siteIdx[j]] #* LGCoeff[siteIdx[j]]
    """
    return nMatrix
    
def CalculateMatrixElements(eigVec, spectrum, LGCoeff, L, basisStrings):
    """
    Calculate the transition matrix elements for a LG beam interacting on the HH lattice.
    ---- Parameters
    ---- Returns
    matrixElementsSquared = square modulus of transition matrix elements: |<Psi_n|O_l|Psi_0>|ˆ2
    """
    print('Calculating matrix elements...')
    Ns = L*L
    Dim = len(basisStrings)
    groundState = eigVec[:,0]
    deltaEnergies = spectrum - spectrum[0]
    n = 0
    tmpTransMatElem = 0
    matElements = np.zeros(len(deltaEnergies), dtype=complex)
    for dE in deltaEnergies:
        print(f'Calculating mat elem <psi_{n}|O|psi_0>')
        if (dE != 0):
            for site in np.arange(0,Ns):
                print(f'creating the LG operator matrix')
                nMat = CreateLGMatrix(site, LGCoeff, basisStrings)
                print(f'computing matrix element')
                tmpMatDotVec = nMat.dot(groundState)
                #if (site == 25):
                    #print(tmpMatDotVec)
                tmpTransMatElem = tmpTransMatElem + ( np.vdot(eigVec[:,n],tmpMatDotVec) )
        matElements[n] = tmpTransMatElem
        tmpTransMatElem = 0
        n = n + 1
        
    #print(deltaEnergies)

    return np.abs(matElements)**2
    
def CalculateMatrixElementsOptimized(eigVec, spectrum, LGCoeff, L, basisStrings, statesWithNonZeroSites):
    """
    Optimize version of CalculateMatrixElements() using CreateLGMatrixOptimized()
    """
    
    print('Calculating matrix elements...')
    Ns = L*L
    Dim = len(basisStrings)
    groundState = eigVec[:,0]
    deltaEnergies = spectrum - spectrum[0]
    n = 0
    tmpTransMatElem = 0
    matElements = np.zeros(len(deltaEnergies), dtype=complex)
    nMat = CreateLGMatrixOptimized(LGCoeff, basisStrings, statesWithNonZeroSites)
    for dE in deltaEnergies:
        print(f'Calculating mat elem <psi_{n}|O|psi_0>')
        if (dE != 0):
            tmpMatDotVec = nMat.dot(groundState)
            tmpTransMatElem = ( np.vdot(eigVec[:,n],tmpMatDotVec) )
        matElements[n] = tmpTransMatElem
        tmpTransMatElem = 0
        n = n + 1
        
    #print(deltaEnergies)

    return np.abs(matElements)**2
    

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
parser.add_argument('-n', type=int, default=0, help='Radial order of the Laguerre polynomials used in the spatial mode of the LG beam')
parser.add_argument('--conf', type=float, help='harmonic trap confinement strength (v0) as v0 * r^2')
parser.add_argument('--alpha', type=float, help='magnetic flux density as alpha=p/q')
parser.add_argument('--hardcore', type=int, nargs='?', const=1, default=0, help='hardcore bosons mode')
parser.add_argument('--nbreigenstates', type=int, help='number of eigenstates to be considered in the matrix elements')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U3 = args.U
if args.U3 is not None: U3 = args.U3
if args.r0 is not None: r0 = args.r0
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf
if args.nbreigenstates is not None: nbrEigenstate = args.nbreigenstates

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True

# Laser parameters
nLG = args.n
print(f'r_0 = {r0}')
print(f'LG radial order n={nLG}')

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

# Load the Hamiltonian at time t=0
#fileName = HHModule.GenFilename(True, L, J, U, trapConf, 2, 0, hamiltonian=True)
#Hamiltonian = LoadHamiltonian(fileName)

# A tuple of (idx for states having non zero elements, idx position of where the non-zero elements are).
#This is needed for the creation of the O_l operator (namely the sum_i n_i operator matrix)
print('Creating the index vectors for the LG operator...')
statesWithNonZeroSites = np.nonzero(basisVectors)
#print(statesWithNonZeroSites)

# Load the eigenvectors
print('Loading the eigenvectors...')
eigVec = np.zeros((Dim,nbrEigenstate), dtype=complex)
for d in np.arange(0,nbrEigenstate):
    fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, 2, d, U3=U3, alpha=FluxDensity, N=N)
    eigVec[:,d] = HHModule.LoadVector(fileName)

print('Loading the energy spectrum...')
fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, 2, 0, spectrum=True, U3=U3, alpha=FluxDensity, N=N)
ESpectrum = LoadSpectrum(fileName)

print('Running the LG excitations...')
for angMom in np.arange(-1,-10,-1):
    print(f'Angular momentum l={angMom}')
    LGCoeff = GenerateLaguerreGauss(L, r0, angMom, nLG)
    matElements = CalculateMatrixElementsOptimized(eigVec, ESpectrum, LGCoeff, L, basisVectors, statesWithNonZeroSites)
    fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, 2, 0, absSpectrum=True, U3=U3, alpha=FluxDensity, N=N)
    SaveAbsorptionSpectrum(fileName + f'_r0_{r0}_neg', ESpectrum-ESpectrum[0], angMom, matElements)
    
for angMom in np.arange(1,10,1):
    print(f'Angular momentum l={angMom}')
    LGCoeff = GenerateLaguerreGauss(L, r0, angMom)
    matElements = CalculateMatrixElementsOptimized(eigVec, ESpectrum, LGCoeff, L, basisVectors, statesWithNonZeroSites)
    fileName = HHModule.GenFilename(hardcore, L, J, U, trapConf, 2, 0, absSpectrum=True, U3=U3, alpha=FluxDensity, N=N)
    SaveAbsorptionSpectrum(fileName + f'_r0_{r0}_pos', ESpectrum-ESpectrum[0], angMom, matElements)

