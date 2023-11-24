import numpy as np
import scipy as sp
import sys
import argparse
import time
import math
import HofstadterThreeBody as HHModule
import GenericModule as gm

def CompatibleStatesWithoutNorm(vectorA, basisVectorsB):
    """
        The same as CalcCompatibleStates() but without calculating the norm factors
    """
    compatibleVectors = []
    for vecB in basisVectorsB:
        motherVec = list(np.array(vectorA) + np.array(vecB))
        compatibleVectors.append(motherVec)
        
    return compatibleVectors
    
def CalcCompatibleStates(vectorA, basisVectorsB):
    """
    Given a state in the reduced Hilbert space H_A we calculate the states in the full Hilbert space H_A + H_B
    that give the coefficients to the new reduce vector.
    """
    compatibleVectors = []
    normFactors = []
    for vecB in basisVectorsB:
        motherVec = list(np.array(vectorA) + np.array(vecB))
        compatibleVectors.append(motherVec)
        currNormFactor = FindNormFactor(motherVec, basisVectorsB)
        normFactors.append(currNormFactor)
        #print(motherVec)
        #print(currNormFactor)
        
    return compatibleVectors, normFactors
    
def FindNormFactor(motherVec, basisVectorsB):
    """
    Find the normalization factor in front of the Schmidt decomposition
    """
    normFactor = 0
    # take the subset of basis vectors that has non zero elements only starting from the position
    # in which the mother vector has non-zero elements
    for vecB in basisVectorsB:
        resVec = np.array(motherVec) - np.array(vecB)
        #print(f'sottrazione {np.array(motherVec)} - {np.array(vecB)}')
        #print(resVec)
        negElems = np.where(resVec < 0)[0]
        #print('np.where')
        #print(np.where(resVec < 0)[0])
        if len(negElems) == 0:
            #print('Buono! norm+1')
            normFactor = normFactor + 1
    
    return normFactor
    
def FindCombinations(N, current_combination=None, start=1):
    """
    We use this for FindNormFactorsByDecomposition
    """
    if current_combination is None:
        current_combination = []

    combinations = []

    for i in range(start, N + 1):
        if i == N:
            combinations.append(current_combination + [i])
        elif i < N:
            combinations += FindCombinations(N - i, current_combination + [i], i)  # Start the inner loop from i + 1

    return combinations
    
def FindNormFactorsByDecomposition(N, Ns, basisVectorsB):
    """
    To optimize the search of the norm factor for every Schmidt decomposition we use this:
    given the total number of particle e.g. N=3 we decompose it into [(3),(2,1),(1,1,1)]
    and we find the norm factors only for these three.
    It returns a dictionary with the different normalization factors.
    """
    combs = FindCombinations(N)
    combLengths = [len(elem) for elem in combs]
    combMotherVectors = [[0] * (Ns - combLengths[i]) + combs[i] for i in np.arange(0,len(combLengths))]
    normFactors = []
    for vec in combMotherVectors:
        norm = FindNormFactor(vec, basisVectorsB)
        normFactors.append(norm)
        
    combs = [tuple(elem) for elem in combs]
    dictionaryNorm = dict(zip(combs,normFactors))
    
    return dictionaryNorm
    

def ReshapeState(basisVectorsA, basisVectorsB, targetState):
    """
    reshape the state |psi> with the same order of the states in the reduced Hilbert space H_B
    """
    psiSectors = [np.zeros((len(basisVectorsA)),dtype=complex) for i in np.arange(0,len(basisVectorsB))]
    c = 0
    for vecB in basisVectorsB:
        compatibleStates, normFactors = CalcCompatibleStates(vecB, basisVectorsA)
        intCompatibleStates = HHModule.PomeranovTag(compatibleStates)
        psiSectors[c][np.arange(0,len(basisVectorsA))] = targetState[np.array(intCompatibleStates)-1]/np.sqrt(normFactors)
        c = c + 1
        
    return psiSectors
  
"""
# We can calculate all the compatibleStates here in a function beforehand
# and put them in a dictionary {Pomeranov(vecB):[array of compatible states]}
def CompatibleStatesDictionary(vectorA, basisVectorsB):
    compatibleVectors = []
    for vecB in basisVectorsB:
        motherVec = list(np.array(vectorA) + np.array(vecB))
        compatibleVectors.append(motherVec)
        """

def ReshapeStateOptimized(basisVectorsA, basisVectorsB, dictionaryNorm, targetState):
    print('Reshaping the state |psi> (ordering it with respect to H_B)...')
    psiSectors = [np.zeros((len(basisVectorsA)),dtype=complex) for i in np.arange(0,len(basisVectorsB))]
    c = 0
    for vecB in basisVectorsB:
        compatibleStates = CompatibleStatesWithoutNorm(vecB, basisVectorsA)
        stateTuples = [tuple(filter(lambda x: x != 0, motherVec)) for motherVec in compatibleStates]
        stateTuples = [tuple(sorted(stateTuple)) for stateTuple in stateTuples]
        normFactors = [dictionaryNorm[stateTuple] for stateTuple in stateTuples]
        intCompatibleStates = HHModule.PomeranovTag(compatibleStates)
        psiSectors[c][np.arange(0,len(basisVectorsA))] = targetState[np.array(intCompatibleStates)-1]/np.sqrt(normFactors)
        c = c + 1
        
    return psiSectors
    
def TraceOutAFromVector(psiSectors, dimA, dimB):
    """
    Build the reduced density matrix rho_A directly from the reshaped vector |psi>
    """
    rho_B = np.zeros((dimB,dimB), dtype=complex)
    for i in np.arange(0,dimB):
        for j in np.arange(0,dimB):
            rho_B[i,j] = np.sum( psiSectors[i]*(psiSectors[j].conj()) )
            
    return rho_B
            
    
def BuildDensityMatrix(psiSectors):
    """
    build the diagonal blocks of the density matrix rho = |psi><psi| (blocks related to the Hilbert space H_B)
    """
    print('Building the density matrix rho=|psi><psi|...')
    rhoBlocks = [np.zeros((np.shape(psiSectors)[1],np.shape(psiSectors)[1]),dtype=complex) for i in np.arange(0,np.shape(psiSectors)[0])]
    for c in np.arange(0,np.shape(psiSectors)[0]):
        rhoBlocks[c] = np.kron(psiSectors[c].T.conj(), psiSectors[c])
        #sparsity = 1.0 - ( np.count_nonzero(rhoBlocks[c]) / float(rhoBlocks[c].size) )
        #print(f'sparsity of the block {c}={sparsity}')
    
    return rhoBlocks

def PartialTrace(rhoBlocks, psiSectors):
    """
    perform the partial trace of rho over one particle subspace A or B and return the reduced density matrix
    """
    print('Performing the partial trace of rho blocks (vectors)...')
    print(f'Shape of the rhoBlocks={np.shape(rhoBlocks)}')
    rhoReduced = np.sum(rhoBlocks, axis=0)
    
    print(f'Shape of the reduced rho vector={np.shape(rhoReduced)}')
    
    print(f'Reshaping the reduced rho block to a matrix (shape={np.shape(psiSectors)[1]}x{np.shape(psiSectors)[1]})...')
    rhoReduced = rhoReduced.reshape((np.shape(psiSectors)[1],np.shape(psiSectors)[1]))
    
    return rhoReduced
    
def EntanglementSpectrum(rhoReduced, nbrEig=30):
    """
    it diagonalizes the reduced density matrix and returns the entanglement spectrum
    """
    print('Diagonalizing rho...')
    spectrum = sp.linalg.eigvalsh(rhoReduced)
    
    return -2*np.log(spectrum)

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
parser.add_argument('--neigenstate', type=int, help='index of the eigenstate for which the PES (particle entanglement spectrum) has to be calculated (e.g. 0 is the groundstate)')
parser.add_argument('--npartition', type=int, help='number of particles to put in the partition N_A, such that N_A + N_B = N')
args = parser.parse_args()

if args.N is not None: N = args.N
if args.L is not None: L = args.L
if args.J is not None: J = args.J
if args.U is not None: U = args.U
if args.U3 is not None: U3 = args.U3
if args.alpha is not None: FluxDensity = args.alpha
if args.conf is not None: trapConf = args.conf
if args.neigenstate is not None: nEigenstate = args.neigenstate

gamma = args.gamma

if args.hardcore == 0:
    hardcore = False
elif args.hardcore == 1:
    hardcore = True
    
if (hardcore == True):
    sys.exit('Particle entanglement spectrum available only in softcore mode')

N_A = args.npartition
N_B = N - N_A

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

# Generate the subspace with N_A particles
if (hardcore == True):
    # Generate the hardcore basis
    print(f'Generating the subspace with {N_A} particles (hardcore)')
    subBasisStringsA = [string for string in HHModule.GenerateHardcoreBasis(N_A, Ns)]
    subBasisVectorsA = np.array([ [int(bit) for bit in string] for string in subBasisStringsA ])
    subIntBasisVectorsA = HHModule.AssignTagToState(subBasisVectorsA, 0)
    subDimA = len(subIntBasisVectorsA)
else:
    print(f'Generating the subspace with {N_A} particles (softcore)')
    # Generate the softcore basis
    subBasisStringsA = [string for string in HHModule.GenerateBasis(N_A, Ns)]
    #print(basisStrings)
    subBasisVectorsA = np.array([ [int(bit) for bit in string] for string in subBasisStringsA ])
    subIntBasisVectorsA = HHModule.PomeranovTag(subBasisVectorsA)
    #print(intBasisVectors)
    subDimA = len(subIntBasisVectorsA)
    
print(f'Partition H_A dim={subDimA}')

# Generate the subspace with N_B particles
print(f'Generating the subspace with {N_B} particles (softcore)')
# Generate the softcore basis
subBasisStringsB = [string for string in HHModule.GenerateBasis(N_B, Ns)]
#print(basisStrings)
subBasisVectorsB = np.array([ [int(bit) for bit in string] for string in subBasisStringsB ])
subIntBasisVectorsB = HHModule.PomeranovTag(subBasisVectorsB)
#print(subIntBasisVectorsB)
subDimB = len(subIntBasisVectorsB)

print(f'Partition H_B dim={subDimB}')

dictionaryNorm = FindNormFactorsByDecomposition(N, Ns, subBasisVectorsB)
print('Norm factors:')
print(dictionaryNorm)

# Load the target state to calculate the entanglement
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N)
targetState = gm.LoadVector(fileName)

start = time.time()
#psiSectors = ReshapeState(subBasisVectorsA, subBasisVectorsB, targetState)
psiSectors = ReshapeStateOptimized(subBasisVectorsA, subBasisVectorsB, dictionaryNorm, targetState)

"""
rhoBlocks = BuildDensityMatrix(psiSectors)
rhoReduced = PartialTrace(rhoBlocks, psiSectors)
gm.TimePrint(start)
if np.shape(rhoReduced)[0] < 10:
    print(rhoReduced)
print(f'Trace = {np.trace(rhoReduced)}')
print('------------------------')
print(f'Purity = {(rhoReduced@rhoReduced).diagonal().sum()}')

spectrum = EntanglementSpectrum(rhoReduced)
print('------------------------')
print('Entanglement spectrum')
print(spectrum)
gm.TimePrint(start)
"""

start = time.time()
rhoReduced = TraceOutAFromVector(psiSectors, subDimA, subDimB)
print(f'Trace = {np.trace(rhoReduced)}')
spectrum = EntanglementSpectrum(rhoReduced)
spectrum.sort()
print('------------------------')
print('Entanglement spectrum')
print(spectrum)
gm.TimePrint(start)

# Save the spectrum
fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N) + '_entanglement_spec'
gm.SaveOneColFile(fileName, spectrum[:30])



