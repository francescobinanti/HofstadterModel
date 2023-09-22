import numpy as np
import scipy as sp
from scipy import special
from scipy.sparse import linalg
import sys
import getopt
import argparse
import time
import bisect
from functools import reduce
import GenericModule as gm

np.set_printoptions(threshold=sys.maxsize)

def TimePrint(start):
    """
    Print the time elapsed in a readable way
    """
    end = time.time()
    dt = end - start
    dtMin = dt/60.
    print('-------- Time elapsed --------')
    print(f'sec={dt}')
    print(f'min={dtMin}')
    print('------------------------------')

def GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, hamiltonian=False, spectrum=False, absSpectrum=False, localDensity=False, c4=False, U3=0.0, alpha=0.0, N=0):
    """
    Returns the filename string the saved eigenstates will have
    """
    tmpString = ""
    fileName = f'bosons_'
    
    if (hardcore == True):
        tmpString = f'hardcore_'
        fileName = fileName + tmpString
    else:
        tmpString = f'softcore_'
        fileName = fileName + tmpString
        
    if (c4 == True):
        tmpString = f'c4_'
        fileName = fileName + tmpString
        
    tmpString = f'N_{N}_x_{L}_y_{L}_J_{J}_alpha_{alpha}_'
    fileName = fileName + tmpString
    
    if ((hardcore == False) and (U != 0.)):
        tmpString = f'U_{U}_'
        fileName = fileName + tmpString
    
    if ((hardcore == False) and (U3 != 0.)):
        tmpString = f'U3_{U3}_'
        fileName = fileName + tmpString
    
    if (trapConf != 0):
        tmpString = f'c_{trapConf}_g_{gamma}_'
        fileName = fileName + tmpString
        
    if (hamiltonian == False) and (spectrum == False) and (absSpectrum == False):
        tmpString = f'n_{nEigenstate}'
        fileName = fileName + tmpString
        
    if (hamiltonian == True):
        tmpString = f'hamiltonian'
        fileName = fileName + tmpString
        
    if (spectrum == True):
        tmpString = f'spectrum'
        fileName = fileName + tmpString
        
    if (absSpectrum == True):
        tmpString = f'absorption'
        fileName = fileName + tmpString
        
    if (localDensity == True):
        tmpString = f'density'
        fileName = fileName + tmpString
        
    return fileName
    
def SaveSpectrum(fileName, energies):
    fileName = fileName + '.dat'
    fileDesc = open(fileName,"a")
    
    for E in energies:
        fileDesc.write(f'{E}\n')
        
def SaveC4Spectrum(fileName, c4Sector, energies):
    """
    Save the absorption spectrum {energy -- angular momentum -- matrix element} on a file
    """
    fileName = fileName + '.dat'
    
    c4Set = [c4Sector for i in np.arange(0,len(energies))]
    dataStack = np.column_stack((c4Set, energies))
    
    print(f'Saving the C4 energy spectrum for L={c4Sector} on file "{fileName}"...')
    with open(fileName,"ab") as fileDesc:
        np.savetxt(fileDesc, dataStack)

def SaveVector(fileName, eigenstate):
    """
    Routine to save eigenstates in .npy format (binary files)
    """
    np.save(fileName, eigenstate)
    
def LoadVector(fileName):
    """
    Routine to load data of an eigenstate from a binary file
    """
    fileName = fileName + '.npy'
    vector = np.load(fileName)
    
    return vector
    
def SaveMatrix(fileName, matrix):
    """
    Save a sparse matrix into a file
    """
    sp.sparse.save_npz(fileName,matrix)

def FindCenter(L):
    """
    Return the center coordinate of a square lattice sized LxL
    """
    return (L/2. - 0.5)

def Phase(ang):
    """
    Return a complex phase factor e^{i*ang}
    """
    return np.exp(complex(0.,ang))
    
def Radius(i, j, c):
    """
    Return the value of radius related to the coordinates (i,j) on the lattice
    """
    return np.sqrt((i-c)*(i-c) + (j-c)*(j-c))

def BitArrayToString(array):
    """
    Transform an array of bits in a string of bits
    """
    resString = ''
    for b in array:
        resString = resString + str(b)
 
    return resString
    
def BitStringToArray(string):
    """
    Transform a string of bit in an array
    """
    return [int(bit) for bit in string]
    
def IntToBinary(intNum, Nb):
    """
    Transform an integer into a binary string of Nb bits
    """
    binary = '{0:0' + str(Nb) + 'b}'
    return binary.format(intNum)
    
def IntToBinaryArray(intNum, Nb):
    """
    Transform an integer into an array of bits
    """
    bitString = IntToBinary(intNum, Nb)
    return BitStringToArray(bitString)
    
def AssignTagToState(basis, n=0):
    """
    It takes the basis vectors and assign a decimal number (a tag in general) to them
    """
    if (n != 1):
        arrayOfStrings = [BitArrayToString(vec) for vec in basis]
        return [int(binString, 2) for binString in arrayOfStrings]
    else:
        binaryString = BitArrayToString(basis)
        return int(binaryString, 2)
        
def RotVec(state, r=1):
    """
    Rotate a state vector by pi/2
    ---- parameters
    state = state vector to be rotated
    r = number of rotations to be performed
    ---- return
    rotVec = array rotated by 90 degrees
    """
    L = int(np.sqrt(len(state)))
        #print('Initial state:')
        #print(state)
    stateMat = np.array([[bit for bit in state[((i-1)*L):(i*L)]] for i in np.arange(L,0,-1)])
        #print('Initial state mat:')
        #print(stateMat)
    rotMat = stateMat.copy()
    for rot in range(r):
        rotMat = np.rot90(rotMat)
        #print('Rotated mat:')
        #print(rotMat)
    rotVec = [rotMat[-(idx//L + 1)][idx%L] for idx in np.arange(0,L*L)]
        #print('Rotated vector:')
        #print(rotVec)
    return rotVec
    
def RotVecInt(stateTag, Ns, softcoreBasis=None, hardcore=True):
    """
    Same as RotVec() but this function works with tags of the states, i.e. it takes an integer that
    represents the state, rotates this latter and gives the integer tag of the rotated state.
    ---- parameters
    stateTag = integer representative of the state
    Ns = number of lattice sites
    softcoreBasis = the basis vectors in the soft-core mode (set only if hardcore=False)
    hardcore = hardcore bosons flag
    --- returns
    rotatedTag = integer representative of the 90-degrees rotated state
    """
    if hardcore == True:
        state = IntToBinaryArray(stateTag, Ns)
        rotState = RotVec(state)
        rotatedTag = AssignTagToState(rotState, n=1)
    else:
        state = softcoreBasis[stateTag-1]
        rotState = RotVec(state)
        rotatedTag = PomeranovTag([rotState])
    
    return rotatedTag
    
def FindRotatedIndex(index, L, r=1):
    """
    Take a spatial index and find the corresponding one after r rotation of the lattice
    """
    stateMat = np.array([[bit for bit in state[((i-1)*L):(i*L)]] for i in np.arange(L,0,-1)])
    #print('Initial state mat:')
    #print(stateMat)
    rotMat = stateMat.copy()
    for rot in range(r):
        rotMat = np.rot90(rotMat)
    #print('Rotated mat:')
    #print(rotMat)
    
    return
    
    
def findIntRep(state, softcoreBasis=None, hardcore=True):
    """
    Take a binary string and find the integer representative of the state under C4 rotation, i.e. apply
    rotation at most 4 times (then the state comes back to itself) and find the minimum integer associated
    ---- parameters
    state = binary array of the initial state for which we want to find the representative
    ---- returns
    repIntState = integer representative of the C4 symmetric state
    pIdx = number of rotations needed to reach the state passed as parameter, from the representative
    """
    Ns = len(state)
    if hardcore == True:
        initStateInt = AssignTagToState(state, n=1)
        rotState = state
        rotStateInt = 0
        repIntState = initStateInt
        while (rotStateInt != initStateInt):
            rotState = RotVec(rotState, r=1)
            rotStateInt = AssignTagToState(rotState, n=1)
            if (rotStateInt < repIntState):
                repIntState = rotStateInt
            
        pIdx = StateDistance(repIntState, initStateInt, Ns)
        
    else:
        initStateInt = PomeranovTag([state])
        rotState = state
        rotStateInt = 0
        repIntState = initStateInt
        while (rotStateInt != initStateInt):
            rotState = RotVec(rotState, r=1)
            rotStateInt = PomeranovTag([rotState])
            if (rotStateInt < repIntState):
                repIntState = rotStateInt
            
        pIdx = StateDistance(repIntState, initStateInt, Ns, softcoreBasis=softcoreBasis, hardcore=False)
            
    return repIntState, pIdx
    
def StateDistance(stateInt1, stateInt2, Ns, softcoreBasis=None, hardcore=True):
    """
    Calculate how many C4 rotations are needed to move from state1 to state2
    """
    dist = 0
    if hardcore == True:
        state1 = BitStringToArray(IntToBinary(stateInt1, Ns))
        state2 = BitStringToArray(IntToBinary(stateInt2, Ns))
        rotState = state1
        rotStateInt = stateInt1
        while (rotStateInt != stateInt2):
            rotState = RotVec(rotState, r=1)
            rotStateInt = AssignTagToState(rotState, n=1)
            dist = dist + 1
            
    else:
        rotStateInt = stateInt1
        while (rotStateInt != stateInt2):
            rotStateInt = RotVecInt(rotStateInt, Ns, softcoreBasis=softcoreBasis, hardcore=False)
            dist = dist + 1
    
    return dist
    
def CalcHofstadterPhaseFromRotation(binState, L, phaseMatrix):
    """
    Calculate the phases e^{i*phi*(x*y)} to attach to a MB state in order to make it C4 symmetric
    in the Hofstadter model.
    We take one generic state, find its rep, then act on the rep with rotations by attaching everytime
    a phase e^{i*phi*(x*y)} (where x,y are the coordinate of the state we are acting on) for each particle,
    until we reach the starting state. The final phase reached is the one of the state passed as parameter.
    """
    totalPhase = 0
    Ns = L*L
    c = FindCenter(L)
    binStateString = BitArrayToString(binState)
    binStateInt = AssignTagToState(binState, n=1)
    repStateInt, idx = findIntRep(binState)
    repStateBinArray = IntToBinary(repStateInt, Ns)
    repStateBinString = BitArrayToString(repStateBinArray)
    #print(f'Representative={repStateBinString}')
    
    nextStateInt=repStateInt
    nextStateArray = repStateBinArray
    nextStateString = repStateBinString
    while (nextStateInt != binStateInt):
        #print(f'Next state in the C4 sector={nextStateString}')
        filledIndices = list(i for i, x in enumerate(nextStateString) if x != '0')
        for idx in filledIndices:
            #print(f'idx={idx}')
            x = idx%L
            y = idx//L
            #totalPhase = totalPhase + (x*y)
            totalPhase = totalPhase + phaseMatrix[x,y]
            #print(f'Adding a phase={(x*y)}')
        
        nextStateArray = RotVec(nextStateArray)
        nextStateString = BitArrayToString(nextStateArray)
        nextStateInt = AssignTagToState(nextStateArray, n=1)
    print(f'Fase che forse sembra vera={totalPhase}')
        
    return totalPhase
    
def EvaluateHofstadterPhaseShift(stateInt, normFactor, L):
    """
    Take a state descriptor and find its phase shift with respect to the representative
    """
    totalPhase = 0
    c = FindCenter(L)
    Ns = L*L
    binState = IntToBinaryArray(stateInt, Ns)
    repStateInt, nbrRotFromRep = findIntRep(binState)
    repStateString = IntToBinary(repStateInt, Ns)
    
    if (nbrRotFromRep != 0):
        # Find the filled indices of the representative and the coordinates, then generate the total phase
        filledIndices = list(i for i, x in enumerate(repStateString) if x != '0')
        for idx in filledIndices:
            x = (idx%L) - c
            y = (idx//L) - c
            totalPhase = totalPhase + ((normFactor - nbrRotFromRep) * x * y)
            #totalPhase = totalPhase + (((Ns-2) - (nbrRotFromRep*L - 1)) * x * y)
        
    return totalPhase

def HilbertDim(N,Ns):
    """
    Calculate the Hilbert space dimension for N particles in Ns sites
    """
    return sp.special.comb(N+Ns-1,N)
    
def MapToOccupations(state):
    """
    Take a Fock state e.g. [3,2,0,0,1] and map it to an array of [m1,m2,...,mN] where
    mi is the site occupied by the i-th particle, e.g. -> [1,1,1,2,2,5]
    """
    occArray = []
    N = np.sum(state)
    filledIndices = list(i for i, x in enumerate(state) if x != '0')
    for idx in filledIndices:
        for n in np.arange(0,state[idx]):
            occArray.append(idx+1)
            
    return occArray
    
def PomeranovTag(arrayOfStates, hardcore=False):
    """
    Assign a tag to a Fock state following Pomeranov algorithm
    """
    N = np.sum(arrayOfStates[0])
    Ns = len(arrayOfStates[0])
    tagArray = []
    if (hardcore == False):
        maxOccupationPerSite = N
    else:
        maxOccupationPerSite = 1
    
    for state in arrayOfStates:
        tmpTag = 1
        occMap = MapToOccupations(state)
        occMap.sort(reverse=True)
        for j in np.arange(1,maxOccupationPerSite+1):
            tmpTag = tmpTag + HilbertDim(j, Ns-occMap[j-1])
        tagArray.append(int(tmpTag))
    
    if len(tagArray) > 1:
        return tagArray
    else:
        return tagArray[0]
    
def GenerateBasis(N, Ns, array=[]):
    """
    Build the MB basis made of strings of dimension Ns, whose sum is constant and equal to N
    ---- parameters
    N = number of particles
    Ns = number of sites
    ---- return
    array of strings describing the MB basis
    """
    if Ns == 0:
        if N == 0:
            yield ''.join(map(str, array))
    elif N < 0:
        return
    else:
        for i in range(N + 1):
            yield from GenerateBasis(N - i, Ns - 1, array + [i])
                
def GenerateHardcoreBasis(N, Ns, array=[]):
    """
    Build the Hardcore MB basis made of strings of dimension Ns, whose sum is constant and equal to N
    ---- parameters
    N = number of particles
    Ns = number of sites
    ---- return
    array of strings describing the MB basis
    """
    if Ns == 0:
        if N == 0:
            yield ''.join(map(str, array))
    elif N < 0:
        return
    else:
        for i in [0, 1]:
            yield from GenerateHardcoreBasis(N - i, Ns - 1, array + [i])
            
def GenerateHardcoreC4Basis(binBasis, L):
    """
    Build the MB reduced Hilbert space related to the C4 sector l
    ---- parameters
    ---- return
    intC4Basis = array of the form [(tag1, n1),(tag2, n2),...] where tag is the min integer of the linear                 combination
                and n is the relative normalization factor for the state.
                pIdx is the phase index related to the Aharonov-Bohm phase, that is acquired in front of
                the state at every rotation.
    """
    Ns = L*L
    intC4Basis = []
    normFactorArray = []
    pIdxArray = []
    n=0
    for state in binBasis:
        #print(f'State n.{n}')
        #n = n+1
        p=0
        normFactor = 0
        initState = state.copy()
        initStateInt = AssignTagToState(initState, n=1)
        rotState = state.copy()
        rotStateInt = 0
        repIntState = initStateInt
        # until it gets to the same state
        while (rotStateInt != initStateInt):
            # rotate and add binStrings to a tmpArray
            rotState = RotVec(rotState, r=1)
            rotStateInt = AssignTagToState(rotState, n=1)
            if (rotStateInt < repIntState):
                repIntState = rotStateInt
                p = p+1
            normFactor = normFactor + 1
        # check if the integer is already inside intC4basis:
                # no -> append the (integer,normFactor) inside intC4Basis
        if repIntState not in intC4Basis:
            intC4Basis.append(repIntState)
            normFactorArray.append(normFactor)
            pIdxArray.append(p)
            
    #print('Array of representative states and their norm factors:')
    #print(intC4Basis)
        
    return [[x,y,z] for x,y,z in zip(intC4Basis, normFactorArray, pIdxArray)] # stack
    
def GenerateSoftcoreC4Basis(binBasis, L):
    """
    Build the MB C4 basis in the soft-core case (i.e. includes Pomeranov indexing)
    """
    Ns = L*L
    intC4Basis = []
    normFactorArray = []
    n=0
    for state in binBasis:
        #print(f'State n.{n}')
        #n = n+1
        normFactor = 0
        initState = state.copy()
        initStateInt = PomeranovTag([initState])
        rotState = state.copy()
        rotStateInt = 0
        repIntState = initStateInt
        # until it gets to the same state
        while (rotStateInt != initStateInt):
            # rotate and add binStrings to a tmpArray
            rotState = RotVec(rotState, r=1)
            rotStateInt = PomeranovTag([rotState])
            
            if (rotStateInt < repIntState):
                repIntState = rotStateInt
                
            normFactor = normFactor + 1
            
        # check if the integer is already inside intC4basis:
        # no -> append the (integer,normFactor) inside intC4Basis
        if repIntState not in intC4Basis:
            intC4Basis.append(repIntState)
            normFactorArray.append(normFactor)
    
    return [[x,y] for x,y in zip(intC4Basis, normFactorArray)]
    
def prodA(indices, state, hardcore = True):
    """
    MB annihilation operator a_i1 a_i2 ... a_in acting from right to left
    ---- parameters
    indices = array of indices (i1,i2,...,in) for the operators
    state = string describing the state we have to act on
    hardcore = flag for hardcore bosons
    ---- return
    coefficient = multiplicative factor after the action of the operators
    finalState = normalized state resulting from the action of the operators ([0,0,...,0] if null state)
    """
    tmpState = state.copy()
    coefficient = 1
    for idx in indices:
        if (hardcore == True):
            if (tmpState[idx] == 0):
                tmpState = [elem-elem for elem in tmpState]
                coefficient = 0
                return tmpState, coefficient
            else:
                tmpState[idx] = 0
        else:
            if (tmpState[idx] == 0):
                tmpState = [elem-elem for elem in tmpState]
                coefficient = 0
                return tmpState, coefficient
            else:
                coefficient = coefficient * np.sqrt(tmpState[idx])
                tmpState[idx] = tmpState[idx] - 1
                
    finalState = tmpState
    
    return finalState, coefficient
    
def prodAd(indices, state, N, hardcore = True):
    """
    MB creation operator ad_i1 ad_i2 ... ad_in acting from right to left
    ---- parameters
    indices = array of indices (i1,i2,...,in) for the operators
    state = string describing the state we have to act on
    hardcore = flag for hardcore bosons
    ---- return
    coefficient = multiplicative factor after the action of the operators
    finalState = normalized state resulting from the action of the operators ([0,0,...,0] if null state)
    """
    tmpState = state.copy()
    coefficient = 1
    for idx in indices:
        if (hardcore == True):
            if (tmpState[idx] == 1):
                tmpState = [elem-elem for elem in tmpState]
                coefficient = 0
                return tmpState, coefficient
            else:
                tmpState[idx] = 1
        else:
            if (np.sum(tmpState)+1 > N):
                tmpState = [elem-elem for elem in tmpState]
                coefficient = 0
                return tmpState, coefficient
            else:
                coefficient = coefficient * np.sqrt(tmpState[idx] + 1)
                tmpState[idx] = tmpState[idx] + 1
    
    finalState = tmpState
    
    return finalState, coefficient
    
def GenLatticeNNLinks(L):
    """
    Generates an array containing the link indices for the nearest neighbors of every site on the lattice.
    E.g. for a 4x4 lattice the bottom-left site is linked upward and on the right, so the first elements
    of the array will be [[(0),1,4], [(1),0,2,5], ...]
    """
    linksArrayVer = [[[] for j in range(L)] for i in range(L)]
    linksArrayHor = [[[] for j in range(L)] for i in range(L)]
    
    for i in range(L):
        for j in range(L):
            if i > 0:
                linksArrayVer[i][j].append((i - 1)*L + j)
            if i < L - 1:
                linksArrayVer[i][j].append((i + 1)*L + j)
            if j > 0:
                linksArrayHor[i][j].append(i*L + (j-1))
            if j < L - 1:
                linksArrayHor[i][j].append(i*L + (j+1))
                
    return linksArrayVer, linksArrayHor
    
def GenLatticeNNLinksOptimized(L):
    """
    Optimized version of GenLatticeNNLinks()
    ---- optimization
    It generates one array only, containing all the horizontal and vertical NN on the lattice.
    It is useful to build the Hamiltonian faster.
    """
    linksArray = [[[] for j in range(L)] for i in range(L)]
    
    for i in range(L):
        for j in range(L):
            if i > 0:
                linksArray[i][j].append((i - 1)*L + j)
            if i < L - 1:
                linksArray[i][j].append((i + 1)*L + j)
            if j > 0:
                linksArray[i][j].append(i*L + (j-1))
            if j < L - 1:
                linksArray[i][j].append(i*L + (j+1))
                
    return linksArray
    
def GetTunnelingDirection(initialState, finalState):
    """
    Take two states and returns the spatial indices related to the particle moving from
    the initial state to the final state, e.g. [0,1,2,0],[1,1,1,0] --> (2,0)
    """
    initialState = np.array(initialState)
    finalState = np.array(finalState)
    # where XOR op. gets zero means that nothing has changed
    xorState = initialState^finalState
    indices = np.nonzero(xorState)[0]
    if (finalState[indices[0]] > initialState[indices[0]]):
        destinationIdx = indices[0]
        startingIdx = indices[1]
    else:
        destinationIdx = indices[1]
        startingIdx = indices[0]
    
    return startingIdx, destinationIdx
    
def BuildHOneBody(binBasis, intBasis, linksVer, linksHor, J, FluxDensity, confinement, gamma=2, debug=False):
    """
    Calculate all the non-zero matrix elements for the hopping terms of the Hofstadter model
    ---- Parameters
    binBasis =
    intBasis =
    """
    print("Building the Hamiltonian...")
    L = len(linksVer)
    D = len(intBasis)
    c = FindCenter(L)
    Hmat = sp.sparse.csc_array((D,D), dtype=complex)
    n=0
    goodState = False
    timeStart = time.time()
    timeStartStates = time.time()
    
    for state in binBasis:
        stateInt = AssignTagToState(state, n=1)
        #timeFor = time.time()
        for i in np.arange(0,L):
            for j in np.arange(0,L):
                idx = i*L + j
                #print("Spanning in vertical links:")
                #print(linksVer[i][j])
                # Vertical hopping (complex conjugate is the same if no phase factor)
                if(state[idx] != 1):
                    for nnIndex in linksVer[i][j]:
                        if (state[nnIndex] != 0):
                            aState = prodA([nnIndex], state)
                            if(state[idx] != 1):
                                adaState = prodAd([idx], aState)
                                adaStateInt = AssignTagToState(adaState, n=1)
                                rowH = intBasis.index(adaStateInt)
                                colH = intBasis.index(stateInt)
                                Hmat[rowH,colH] += -J
                                #print("Non-zero matrix element initial state (Vertical):")
                                #print(state)
                                #del adaState, adaStateInt
                                goodState = True
                    # Horizontal hopping
                    for nnIndex in linksHor[i][j]:
                        if (state[nnIndex] != 0):
                            aState = prodA([nnIndex], state)
                            if(state[idx] != 1):
                                adaState = prodAd([idx], aState)
                                adaStateInt = AssignTagToState(adaState, n=1)
                                rowH = intBasis.index(adaStateInt)
                                colH = intBasis.index(stateInt)
                                if (nnIndex > idx): # left-hopping
                                    Hmat[rowH,colH] += -J*Phase(-2.*np.pi*FluxDensity*i)
                                else: # right-hopping (complex-conjugate)
                                    Hmat[rowH,colH] += -J*Phase(2.*np.pi*FluxDensity*i)
                                #del adaState, adaStateInt
                                goodState = True
                # Trap confinement
                if (state[idx] != 0):
                    idxDiagH = intBasis.index(stateInt)
                    Hmat[idxDiagH,idxDiagH] += state[idx] * confinement * (Radius(i, j, c) ** gamma)
        #endTimeFor = time.time()
        #print(f'Time for the for loop over sites t={(endTimeFor-timeFor)}')
        if (goodState == True):
            n = n+1
            if (n%1000 == 0):
                timeEnd = time.time()
                print("basis state n."+str(n)+"/"+str(D))
                locTime = (timeEnd - timeStartStates)
                print(f'time for 1000 states: {locTime/60.:.2f} mins , {locTime:.2f} sec')
                totTime = (timeEnd - timeStart)
                print(f'total time: {totTime/60.:.2f} mins , {totTime:.2f} sec')
                print(f'Expected time for building H: {D/n * (totTime/60.):.2f} mins , {D/n * (totTime):.2f} sec')
                timeStartStates = time.time()
                print("-----")
            goodState = False
        
    if (debug == True):
        print(Hmat.toarray())
        
    print("Number of states (off-diag) contributing to the Hamiltonian: " + str(n))

    return Hmat
    
def BuildHOneBodyOptimized(binBasis, intBasis, links, J, FluxDensity, confinement, gamma=2, hardcore=True):
    """
    Optimized version of BuildHOneBody()
    ---- optimization
    1) It takes the MB state and transform it into a string, such that we spot the indices
        related to the presence of '1's in the string, that will be our variable 'idx'.
        In this way we don't have to iterate the sites with the nested for loops.
    2) Use the bisection to find the indices related to the integer tag of the states
    3) Use only one array 'links' and distinguish the horizontal hopping by checking
        if 'nnIndex = idx +- 1'
    """
    print("Building the Hamiltonian (optimized)...")
    L = len(links)
    D = len(intBasis)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    n = 0
    colH = 1
    
    for state in binBasis:
        #print(f'State n.{n}')
        #n = n+1
        stateInt = AssignTagToState(state, n=1)
        # this is the ket of H|state>
        stateString = BitArrayToString(state)
        # find the sites (idx) in which we don't have a particle
        indices = list(i for i, x in enumerate(stateString) if x == '0')
        for idx in indices:
            for nnIndex in links[idx//L][idx%L]:
                if (state[nnIndex] != 0):
                    aState, coeffA = prodA([nnIndex], state)
                    if(state[idx] != 1):
                        adaState, coeffAdA = prodAd([idx], aState, N)
                        adaStateInt = AssignTagToState(adaState, n=1)
                        rowH = bisect.bisect_right(intBasis,adaStateInt)
                        if (nnIndex == (idx+1)): # left-hopping
                            Hmat[rowH-1,colH-1] += -J*Phase(-2.*np.pi*FluxDensity*(idx//L))
                            #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J*Phase(-2.*np.pi*FluxDensity*(idx//L))}')
                        elif (nnIndex == (idx-1)): # right-hopping
                            Hmat[rowH-1,colH-1] += -J*Phase(2.*np.pi*FluxDensity*(idx//L))
                            #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J*Phase(2.*np.pi*FluxDensity*(idx//L))}')
                        else:
                            Hmat[rowH-1,colH-1] += -J
                            #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J}')
        colH = colH + 1
        
        # add the confinement trap
        filledSites = list(i for i, x in enumerate(stateString) if x == '1')
        for idx in filledSites:
            idxDiagH = bisect.bisect_right(intBasis, stateInt)
            i = idx//L
            j = idx%L
            Hmat[idxDiagH-1,idxDiagH-1] += state[idx] * confinement * (Radius(i, j, c) ** gamma)
            
    return Hmat
    
def BuildHOneBodyOptimizedSoftCore(binBasis, intBasis, links, J, FluxDensity, confinement, gamma=2):
    """
    Optimized version of BuildHOneBody() for soft-core bosons
    """
    print("Building the soft-core Hamiltonian...")
    L = len(links)
    D = len(intBasis)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    print(f'Number of bosons={N}')
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    n = 0
    colH = 1
    
    for state in binBasis:
        #print(f'State n.{n}')
        #n = n+1
        stateInt = PomeranovTag([state])
        # this is the ket of H|state>
        stateString = BitArrayToString(state)
        # find the sites (idx) in which we don't have a particle
        indices = list(i for i, x in enumerate(stateString) if x != '0')
        for idx in indices:
            aState, coeffA = prodA([idx], state, hardcore=False)
            for nnIndex in links[idx//L][idx%L]:
                if (state[nnIndex] < N):
                    adaState, coeffAdA = prodAd([nnIndex], aState, N, hardcore=False)
                    #adaStateInt = AssignTagToState(adaState, n=1)
                    #rowH = bisect.bisect_right(intBasis,adaStateInt)
                    rowH = PomeranovTag([adaState])
                    if (nnIndex == (idx+1)): # right-hopping
                        Hmat[rowH-1,colH-1] += -J*Phase(2.*np.pi*FluxDensity*(idx//L))*coeffA*coeffAdA
                        #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J*Phase(-2.*np.pi*FluxDensity*(idx//L))}')
                    elif (nnIndex == (idx-1)): # left-hopping
                        Hmat[rowH-1,colH-1] += -J*Phase(-2.*np.pi*FluxDensity*(idx//L))*coeffA*coeffAdA
                        #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J*Phase(2.*np.pi*FluxDensity*(idx//L))}')
                    else:
                        Hmat[rowH-1,colH-1] += -J*coeffA*coeffAdA
                        #print(f'Matrix element <{rowH-1}|H|{colH-1}> = {-J}')
        colH = colH + 1
        
        # add the confinement trap
        filledSites = list(i for i, x in enumerate(stateString) if x != '0')
        for idx in filledSites:
            i = idx//L
            j = idx%L
            Hmat[stateInt-1,stateInt-1] += state[idx] * confinement * (Radius(i, j, c) ** gamma)
            
    return Hmat
    
def BuildHOneBodyC4Symmetry(repStatesArray, binBasis, L, links, J, FluxDensity, confinement, gamma, l):
    """
    It builds the hopping Hamiltonian blocks with the C4 symmetry
    """
    print(f'Building the C4 Hamiltonian in sector l={l}')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    # number of particles
    N = np.sum(binBasis[0])
    print(f'Number of particles={N}')
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    tmpMatElem = 0
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    #n=0
    for elem in repStatesArray:
        #print(f'State n.{n}')
        #n = n+1
        repStateInt = elem[0]
        normFactor = elem[1]
        repStateBin = '{0:0' + str(Ns) + 'b}'
        repStateBin = repStateBin.format(repStateInt)
        repState = BitStringToArray(repStateBin)
        # reconstruct the symmetric state from the representative
        for i in np.arange(0,normFactor):
            currSubState = RotVec(repState, r=i)
            currSubStateString = BitArrayToString(currSubState)
            currSubStateInt = AssignTagToState(currSubStateString, n=1)
            #print(f'Acting with H on ket=|{currSubStateString}> (|{AssignTagToState(currSubStateString, n=1)}>)')
            currSubStatePhase = Phase(2.*np.pi*l*i / 4.)
            
            # here the procedure is more or less the same as in BuildHOneBodyOptimized()
            # find the sites (idx) in which we have a particle
            filledIndices = list(i for i, x in enumerate(currSubStateString) if x == '1')
            for idx in filledIndices:
                iCoord = idx//L
                jCoord = idx%L
                aState, coeffA = prodA([idx], currSubState)
                for nnIndex in links[idx//L][idx%L]:
                    iNNCoord = nnIndex//L
                    jNNCoord = nnIndex%L
                    if (currSubState[nnIndex] == 0):
                        adaState, coeffAdA = prodAd([nnIndex], aState, N)
                        # find the representative of this state and the phase related
                        adaStateInt, phaseRowIdx = findIntRep(adaState)
                        if adaStateInt in intRepArray:
                            adaStateString = BitArrayToString(adaState)
                            adaInt = AssignTagToState(adaStateString, n=1)
                            
                            rowH = intRepArray[adaStateInt]
                            normFactorRow = repStatesArray[rowH][1]
                            rowStatePhase = Phase(2.*np.pi*l*phaseRowIdx / 4.)
                                
                            if (nnIndex == (idx+1)): # right-hopping
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                            elif (nnIndex == (idx-1)): # left-hopping
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                            elif (nnIndex > (idx+1)): # upward
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                            elif (nnIndex < (idx-1)): #downward
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                            
                                
                # add confinement
                Hmat[colH,colH] += currSubState[idx] * confinement * (Radius(iCoord, jCoord, c) ** gamma) * (1./normFactor)
        colH = colH + 1
    
    return Hmat
    
def BuildHOneBodyC4SymmetryOptimized(repStatesArray, binBasis, L, links, J, FluxDensity, confinement, gamma, l):
    """
    Optimized version of BuildHOneBodyC4Symmetry()
    --- Optimization
    Now we act ONLY on the representative state (so get rid of one FOR loop inside each C4 state) and then we build the other matrix elements by rotating the
    resulting action on the representative, i.e. [R, a_d a] = 0 (rotation and hopping commute)
    """
    print(f'Building the C4 Hamiltonian (optimized) in sector l={l}')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    # number of particles
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    tmpMatElem = 0
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    #n=0
    for elem in repStatesArray:
        #print(f'State n.{n}')
        #n = n+1
        repStateInt = elem[0]
        normFactor = elem[1]
        repStateString = IntToBinary(repStateInt, Ns)
        repState = BitStringToArray(repStateString)
        #print(f'Rep state={repStateString}')
            
        # here the procedure is more or less the same as in BuildHOneBodyOptimized()
        # find the sites (idx) in which we have a particle
        filledIndices = list(i for i, x in enumerate(repStateString) if x == '1')
        for idx in filledIndices:
            iCoord = idx//L
            jCoord = idx%L
            aState, coeffA = prodA([idx], repState)
            for nnIndex in links[iCoord][jCoord]:
                iNNCoord = nnIndex//L
                jNNCoord = nnIndex%L
                if (repState[nnIndex] == 0):
                    adaState, coeffAdA = prodAd([nnIndex], aState, N)
                    # find the representative of this state and the phase related
                    adaStateInt, phaseRowIdx = findIntRep(adaState)
                    if adaStateInt in intRepArray:
                        #adaStateString = BitArrayToString(adaState)
                        #adaInt = AssignTagToState(adaStateString, n=1)
                        #print(f'Row vector={BitArrayToString(adaState)}')
                        
                        rowH = intRepArray[adaStateInt]
                        normFactorRow = repStatesArray[rowH][1]
                        rowStatePhase = Phase(2.*np.pi*l*phaseRowIdx / 4.)
                            
                        if (nnIndex == (idx+1)): # right-hopping
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                        elif (nnIndex == (idx-1)): # left-hopping
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                        elif (nnIndex > (idx+1)): # upward
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                        elif (nnIndex < (idx-1)): #downward
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                            
                        # Here we rotate the resulting AdA|ket> (and <bra| also) state (at most normFactorRow-phaseRowIdx-1 times) and find the other matrix elements (off-diagonal terms)
                        for nbrRot in np.arange(1,normFactorRow):
                            # The representative of this is still the representative of AdA|state>
                            rotatedAdAKet = RotVec(adaState, r=nbrRot)
                            #print(f'rotation-->{BitArrayToString(rotatedAdAKet)}')
                            rotatedAdAKetInt = AssignTagToState(rotatedAdAKet, n=1)
                            rotatedRowStatePhase = Phase(2.*np.pi*l*(phaseRowIdx+nbrRot) / 4.)
                            # The representative of this is still repState, so no need to rotate it accordingly
                            rotatedKet = RotVec(repState, r=nbrRot)
                            rotatedKetPhase = Phase(2.*np.pi*l*nbrRot / 4.)
                            
                            startIdx, destIdx = GetTunnelingDirection(rotatedKet, rotatedAdAKet)
                            
                            iRotCoord = startIdx//L
                            jRotCoord = startIdx%L
                            
                            if (destIdx == (startIdx+1)):
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iRotCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iRotCoord-c))}')
                            elif (destIdx == (startIdx-1)):
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iRotCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iRotCoord-c))}')
                            elif (destIdx > (startIdx+1)):
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jRotCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jRotCoord-c))}')
                            elif (destIdx < (startIdx-1)):
                                Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jRotCoord-c))
                                #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jRotCoord-c))}')
            
            # add confinement
            Hmat[colH,colH] += repState[idx] * confinement * (Radius(iCoord, jCoord, c) ** gamma) * (1./normFactor)
            
            iPrevCoord = iCoord
            jPrevCoord = jCoord
            # Here we rotate for the diagonal elements (confinement) instead
            for nbrRot in np.arange(1,normFactor):
                rotatedKet = RotVec(repState, r=nbrRot)
                rotatedKetString = BitArrayToString(rotatedKet)
                #rotatedFilledIndices = list(i for i, x in enumerate(rotatedKetString) if x == '1')
                
                # Find the rotated coordinates of the rotated state
                iRotCoord = jPrevCoord
                jRotCoord = -iPrevCoord+(2*c)
                rIdx = jRotCoord + L*iRotCoord
                
                Hmat[colH,colH] += rotatedKet[int(rIdx)] * confinement * (Radius(iRotCoord, jRotCoord, c) ** gamma) * (1./normFactor)
                
                iPrevCoord = iRotCoord
                jPrevCoord = jRotCoord
                    
        colH = colH + 1
    
    return Hmat
    
def BuildHOneBodySoftCoreC4Symmetry(repStatesArray, binBasis, L, links, J, FluxDensity, confinement, gamma, l):
    """
    It builds the hopping Hamiltonian blocks with the C4 symmetry in the soft-core case
    """
    print(f'Building the C4 Hamiltonian in sector l={l}')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    # number of particles
    N = np.sum(binBasis[0])
    print(f'Number of particles={N}')
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    tmpMatElem = 0
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    for elem in repStatesArray:
        repStateInt = elem[0]
        #print(f'Working on the representative={repStateInt}')
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        #print(f'Associated state={repState}')
        # reconstruct the symmetric state from the representative
        for i in np.arange(0,normFactor):
            #print(f'rotating the state {i} times')
            currSubState = RotVec(repState, r=i)
            currSubStateString = BitArrayToString(currSubState)
            #print(f'sub state={currSubStateString}')
            currSubStateInt = PomeranovTag([currSubState])
            #print(f'Acting with H on ket=|{currSubStateString}> (|{AssignTagToState(currSubStateString, n=1)}>)')
            currSubStatePhase = Phase(2.*np.pi*l*i / 4.)
            
            # here the procedure is more or less the same as in BuildHOneBodyOptimized()
            # find the sites (idx) in which we have a particle
            filledIndices = list(i for i, x in enumerate(currSubStateString) if x != '0')
            #print(f'filled indices are: {filledIndices}')
            for idx in filledIndices:
                iCoord = idx//L
                jCoord = idx%L
                #print('Acting with a')
                aState, coeffA = prodA([idx], currSubState, hardcore=False)
                #print(f'resulting state={aState}')
                for nnIndex in links[idx//L][idx%L]:
                    iNNCoord = nnIndex//L
                    jNNCoord = nnIndex%L
                    #if (currSubState[nnIndex] == 0):
                    adaState, coeffAdA = prodAd([nnIndex], aState, N, hardcore=False)
                    # find the representative of this state and the phase related
                    adaStateInt, phaseRowIdx = findIntRep(adaState, softcoreBasis=binBasis, hardcore=False)
                    #print(f'ada |state> = {BitArrayToString(adaState)}')
                    #print(f'rep={adaStateInt}')
                    if adaStateInt in intRepArray:
                        #adaStateString = BitArrayToString(adaState)
                        #adaInt = AssignTagToState(adaStateString, n=1)
                        
                        rowH = intRepArray[adaStateInt]
                        normFactorRow = repStatesArray[rowH][1]
                        rowStatePhase = Phase(2.*np.pi*l*phaseRowIdx / 4.)
                            
                        if (nnIndex == (idx+1)): # right-hopping
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))*coeffA*coeffAdA
                        elif (nnIndex == (idx-1)): # left-hopping
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))*coeffA*coeffAdA
                        elif (nnIndex > (idx+1)): # upward
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))*coeffA*coeffAdA
                        elif (nnIndex < (idx-1)): #downward
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * currSubStatePhase * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))*coeffA*coeffAdA
                            
                                
                # add confinement
                Hmat[colH,colH] += currSubState[idx] * confinement * (Radius(iCoord, jCoord, c) ** gamma) * (1./normFactor)
        colH = colH + 1
    
    return Hmat
    
def BuildHOneBodySoftCoreC4SymmetryOptimized(repStatesArray, binBasis, L, links, J, FluxDensity, confinement, gamma, l):
    """
    Optimized version of BuildHOneBodySoftCoreC4Symmetry() acting ONLY on the representative states
    """
    print(f'Building the C4 Hamiltonian (optimized) in sector l={l}')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    # number of particles
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    tmpMatElem = 0
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    #n=0
    for elem in repStatesArray:
        #print(f'State n.{n}')
        #n = n+1
        repStateInt = elem[0]
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        repStateString = BitArrayToString(repState)
        #print(f'Rep state={repStateString}')
            
        # here the procedure is more or less the same as in BuildHOneBodyOptimized()
        # find the sites (idx) in which we have a particle
        filledIndices = list(i for i, x in enumerate(repStateString) if x != '0')
        for idx in filledIndices:
            iCoord = idx//L
            jCoord = idx%L
            aState, coeffA = prodA([idx], repState, hardcore=False)
            for nnIndex in links[iCoord][jCoord]:
                iNNCoord = nnIndex//L
                jNNCoord = nnIndex%L
                adaState, coeffAdA = prodAd([nnIndex], aState, N, hardcore=False)
                # find the representative of this state and the phase related
                adaStateInt, phaseRowIdx = findIntRep(adaState, softcoreBasis=binBasis, hardcore=False)
                if adaStateInt in intRepArray:
                    #adaStateString = BitArrayToString(adaState)
                    #adaInt = AssignTagToState(adaStateString, n=1)
                    #print(f'Row vector={BitArrayToString(adaState)}')
                    
                    rowH = intRepArray[adaStateInt]
                    normFactorRow = repStatesArray[rowH][1]
                    rowStatePhase = Phase(2.*np.pi*l*phaseRowIdx / 4.)
                        
                    if (nnIndex == (idx+1)): # right-hopping
                        Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))*coeffA*coeffAdA
                        #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                    elif (nnIndex == (idx-1)): # left-hopping
                        Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))*coeffA*coeffAdA
                        #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iCoord-c))}')
                    elif (nnIndex > (idx+1)): # upward
                        Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))*coeffA*coeffAdA
                        #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                    elif (nnIndex < (idx-1)): #downward
                        Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))*coeffA*coeffAdA
                        #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * np.conj(rowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jCoord-c))}')
                        
                    # Here we rotate the resulting AdA|ket> (and <bra| also) state (at most normFactorRow-phaseRowIdx-1 times) and find the other matrix elements (off-diagonal terms)
                    for nbrRot in np.arange(1,normFactorRow):
                        # The representative of this is still the representative of AdA|state>
                        rotatedAdAKet = RotVec(adaState, r=nbrRot)
                        #print(f'rotation-->{BitArrayToString(rotatedAdAKet)}')
                        rotatedAdAKetInt = PomeranovTag([rotatedAdAKet])
                        rotatedRowStatePhase = Phase(2.*np.pi*l*(phaseRowIdx+nbrRot) / 4.)
                        # The representative of this is still repState, so no need to rotate it accordingly
                        rotatedKet = RotVec(repState, r=nbrRot)
                        rotatedKetPhase = Phase(2.*np.pi*l*nbrRot / 4.)
                        
                        startIdx, destIdx = GetTunnelingDirection(rotatedKet, rotatedAdAKet)
                        
                        iRotCoord = startIdx//L
                        jRotCoord = startIdx%L
                        
                        if (destIdx == (startIdx+1)):
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iRotCoord-c))*coeffA*coeffAdA
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(iRotCoord-c))}')
                        elif (destIdx == (startIdx-1)):
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iRotCoord-c))*coeffA*coeffAdA
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(iRotCoord-c))}')
                        elif (destIdx > (startIdx+1)):
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jRotCoord-c))*coeffA*coeffAdA
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(-2.*np.pi*FluxDensity*0.5*(jRotCoord-c))}')
                        elif (destIdx < (startIdx-1)):
                            Hmat[rowH,colH] += -J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jRotCoord-c))*coeffA*coeffAdA
                            #print(f'({rowH},{colH})={-J * (1./np.sqrt(normFactor*normFactorRow)) * rotatedKetPhase * np.conj(rotatedRowStatePhase) * Phase(2.*np.pi*FluxDensity*0.5*(jRotCoord-c))}')
                                
            # add confinement
            Hmat[colH,colH] += repState[idx] * confinement * (Radius(iCoord, jCoord, c) ** gamma) * (1./normFactor)
            
            iPrevCoord = iCoord
            jPrevCoord = jCoord
            # Here we rotate for the diagonal elements (confinement) instead
            for nbrRot in np.arange(1,normFactor):
                rotatedKet = RotVec(repState, r=nbrRot)
                rotatedKetString = BitArrayToString(rotatedKet)
                
                # Find the rotated coordinates of the rotated state
                iRotCoord = jPrevCoord
                jRotCoord = -iPrevCoord+(2*c)
                rIdx = jRotCoord + L*iRotCoord
                
                Hmat[colH,colH] += rotatedKet[int(rIdx)] * confinement * (Radius(iRotCoord, jRotCoord, c) ** gamma) * (1./normFactor)
                
                iPrevCoord = iRotCoord
                jPrevCoord = jRotCoord
                    
        colH = colH + 1
        
    return Hmat
                                

def BuildHTwoBodyOnsite(binBasis, intBasis, L, U):
    """
    Calculate all the non-zero matrix elements for the two-body onsite interaction terms of the Hofstadter model,
    i.e. the terms proportional to U/2 n_i(n_i - 1)
    """
    D = len(intBasis)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    
    
    for state in binBasis:
        stateInt = PomeranovTag([state])
        stateString = BitArrayToString(state)
        
        filledSites = list(i for i, x in enumerate(stateString) if (x != '0' and x != '1'))
        for idx in filledSites:
            Hmat[stateInt-1,stateInt-1] += ( state[idx]*(state[idx] - 1) ) * (U/2.)
    
    return Hmat
    
def BuildHThreeBodyOnsite(binBasis, intBasis, L, U3):
    """
    Build the three-body onsite interaction Hamiltonian: U/6 n_i(n_i - 1)(n_i - 2)
    """
    D = len(intBasis)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    
    
    for state in binBasis:
        stateInt = PomeranovTag([state])
        stateString = BitArrayToString(state)
        
        filledSites = list(i for i, x in enumerate(stateString) if (x != '0' and x != '1' and x != '2'))
        for idx in filledSites:
            Hmat[stateInt-1,stateInt-1] += ( state[idx]*(state[idx] - 1)*(state[idx] - 2) ) * (U3/6.)
    
    return Hmat
    
def BuildHTwoBodyOnsiteC4(repStatesArray, binBasis, L, U):
    """
    Build the two-body onsite interaction hamiltonian in the C4 basis
    """
    print(f'Building the two-body interaction C4 Hamiltonian')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    for elem in repStatesArray:
        repStateInt = elem[0]
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        # reconstruct the symmetric state from the representative
        for i in np.arange(0,normFactor):
            #print(f'rotating the state {i} times')
            currSubState = RotVec(repState, r=i)
            currSubStateString = BitArrayToString(currSubState)
            currSubStateInt = PomeranovTag([currSubState])
            
            filledIndices = list(i for i, x in enumerate(currSubStateString) if x != '0')
            for idx in filledIndices:
                Hmat[colH,colH] += (U/2.) * ( currSubState[idx] * (currSubState[idx] - 1) ) * (1./normFactor)
            
        colH = colH + 1
    
    return Hmat
    
def BuildHTwoBodyOnsiteC4Optimized(repStatesArray, binBasis, L, U):
    """
    Optimized version of BuildHTwoBodyOnsiteC4() by acting ONLY on the representative state
    """
    print(f'Building the two-body interaction C4 Hamiltonian (optimized)')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    for elem in repStatesArray:
        #print(f'State n.{n}')
        #n = n+1
        repStateInt = elem[0]
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        repStateString = BitArrayToString(repState)

        filledIndices = list(i for i, x in enumerate(repStateString) if x != '0')
        for idx in filledIndices:
            iCoord = idx//L
            jCoord = idx%L
            
            Hmat[colH,colH] += (U/2.) * ( repState[idx] * (repState[idx] - 1) ) * (1./normFactor)
            
            iPrevCoord = iCoord
            jPrevCoord = jCoord
            
            for nbrRot in np.arange(1,normFactor):
                rotatedState = RotVec(repState, r=nbrRot)
                rotatedStateString = BitArrayToString(rotatedState)
                
                # Find the rotated coordinates of the rotated state
                iRotCoord = jPrevCoord
                jRotCoord = -iPrevCoord+(2*c)
                rIdx = jRotCoord + L*iRotCoord
                
                Hmat[colH,colH] += (U/2.) * ( rotatedState[int(rIdx)] * (rotatedState[int(rIdx)] - 1) ) * (1./normFactor)
                
                iPrevCoord = iRotCoord
                jPrevCoord = jRotCoord
                    
        colH = colH + 1
        
    return Hmat
    
def BuildHThreeBodyOnsiteC4(repStatesArray, binBasis, L, U3):
    """
    Build the two-body onsite interaction hamiltonian in the C4 basis
    """
    print(f'Building the three-body interaction C4 Hamiltonian')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    tmpMatElem = 0
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    for elem in repStatesArray:
        repStateInt = elem[0]
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        # reconstruct the symmetric state from the representative
        for i in np.arange(0,normFactor):
            #print(f'rotating the state {i} times')
            currSubState = RotVec(repState, r=i)
            currSubStateString = BitArrayToString(currSubState)
            currSubStateInt = PomeranovTag([currSubState])
            
            filledIndices = list(i for i, x in enumerate(currSubStateString) if x != '0')
            for idx in filledIndices:
                Hmat[colH,colH] += (U3/6.) * ( currSubState[idx] * (currSubState[idx] - 1) * (currSubState[idx] - 2) ) * (1./normFactor)
            
        colH = colH + 1
    
    return Hmat
    
def BuildHThreeBodyOnsiteC4Optimized(repStatesArray, binBasis, L, U3):
    """
    Optimized version of BuildHThreeBodyOnsiteC4() by acting ONLY on the representative state
    """
    print(f'Building the two-body interaction C4 Hamiltonian (optimized)')
    Ns = L*L
    D = len(repStatesArray)
    c = FindCenter(L)
    N = np.sum(binBasis[0])
    Hmat = sp.sparse.lil_matrix((D,D), dtype=complex)
    # recall the repStatesArray integers are sorted, so colH=0,1,2,3... as long as we span repStatesArray
    colH = 0
    intRepArray = {subArray[0]: i for i, subArray in enumerate(repStatesArray)}
    for elem in repStatesArray:
        #print(f'State n.{n}')
        #n = n+1
        repStateInt = elem[0]
        normFactor = elem[1]
        repState = binBasis[repStateInt-1]
        repStateString = BitArrayToString(repState)

        filledIndices = list(i for i, x in enumerate(repStateString) if x != '0')
        for idx in filledIndices:
            iCoord = idx//L
            jCoord = idx%L
            
            Hmat[colH,colH] += (U3/6.) * ( repState[idx] * (repState[idx] - 1) * (repState[idx] - 2) ) * (1./normFactor)
            
            iPrevCoord = iCoord
            jPrevCoord = jCoord
            
            for nbrRot in np.arange(1,normFactor):
                rotatedState = RotVec(repState, r=nbrRot)
                rotatedStateString = BitArrayToString(rotatedState)
                
                # Find the rotated coordinates of the rotated state
                iRotCoord = jPrevCoord
                jRotCoord = -iPrevCoord+(2*c)
                rIdx = jRotCoord + L*iRotCoord
                
                Hmat[colH,colH] += (U3/6.) * ( rotatedState[int(rIdx)] * (rotatedState[int(rIdx)] - 1) * (rotatedState[int(rIdx)] - 2) ) * (1./normFactor)
                
                iPrevCoord = iRotCoord
                jPrevCoord = jRotCoord
                    
        colH = colH + 1
        
    return Hmat

    
def diagH(Hmat,nEigenv):
    """
    Diagonalize the sparse matrix Hamiltonian
    ---- parameters
    nEigenv = number of eigenvectors to be computed
    ---- return
    [1] array of eigenvalues
    [2] array of eigenvectors
    """
    print("Diagonalizing the Hamiltonian...")
    return sp.sparse.linalg.eigsh(Hmat, k=nEigenv, tol=1e-14, which='SA')
    
def RotateMatrix(H, arrayBasis):
    """
    Function that rotates the basis of a matrix by 90 degrees and rearrange its elements accordingly
    """
    intBasis = AssignTagToState(arrayBasis, 0)
    Ns = len(intBasis)
    # enumerate the basis in the format {intTag:positionInTheMatrix}
    enumBasis = {intArray: i for i, intArray in enumerate(intBasis)}
    
    # Rotates the elements of the basis
    rotatedBasis = [RotVec(arrayBasis[i]) for i in np.arange(0,Ns)]
    rotatedIntBasis = AssignTagToState(rotatedBasis, 0)
    permutationList = [enumBasis[tag] for tag in rotatedIntBasis]
    
    permutedRows = H[permutationList,:]
    rotatedMatrix = permutedRows[:,permutationList]

    return rotatedMatrix
    
# Only if ran by the current module
if __name__ == "__main__":

    J = 1.
    U = 0.
    U3 = 0.
    FluxDensity = 0.125
    trapConf = 0.

    # Save flags
    saveEigenstates = False
    saveSpectrum = True
    saveHamiltonian = False

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
    parser.add_argument('--savestates', type=int, nargs='?', const=1, default=0, help='save each eigenvector on a file')
    parser.add_argument('--savehamiltonian', type=int, nargs='?', const=1, default=0, help='save the hamiltonian on a file')
    parser.add_argument('--loadhamiltonian', type=int, nargs='?', const=1, default=0, help='load the hamiltonian from a file .npz without building it from scratch')
    parser.add_argument('--nbreigenstates', type=int, help='number of eigenstates to be saved')
    parser.add_argument('--c4symmetry', type=int, nargs='?', const=1, default=0, help='use the c4 rotation symmetry in the exact diagonalization')
    args = parser.parse_args()
    
    if args.N is not None: N = args.N
    if args.L is not None: L = args.L
    if args.J is not None: J = args.J
    if args.U is not None: U = args.U
    if args.U3 is not None: U3 = args.U3
    if args.alpha is not None: FluxDensity = args.alpha
    if args.conf is not None: trapConf = args.conf
    if args.nbreigenstates is not None: nbrEigenstate = args.nbreigenstates
    
    gamma = args.gamma
    
    if args.hardcore == 0:
        hardcore = False
    elif args.hardcore == 1:
        hardcore = True
        
    if args.savestates == 0:
        saveEigenstates = False
    elif args.savestates == 1:
        saveEigenstates = True
        
    if args.c4symmetry == 0:
        c4Flag = False
    elif args.c4symmetry == 1:
        c4Flag = True
        
    if args.savehamiltonian == 0:
        saveHamiltonian = False
    elif args.savehamiltonian == 1:
        saveHamiltonian = True
        
    if args.loadhamiltonian == 0:
        loadHamiltonian = False
    elif args.loadhamiltonian == 1:
        loadHamiltonian = True


    Ns = L*L
    Dim = 0
    c = FindCenter(L)
    
    print(f'N={N}')
    print(f'L={L} --> {L}x{L} lattice')
    print(f'J={J}')
    print(f'Harmonic trap v0={trapConf}')
    print(f'Flux density alpha={FluxDensity}')

    if (hardcore == True):
        print('Hardcore mode')
        print('Generating basis vectors...')
        # Generate the MB basis vectors using generators
        basisVectors = [ [int(bit) for bit in string] for string in GenerateHardcoreBasis(N, Ns) ]
        #print(basisVectors)

        intBasisVectors = AssignTagToState(basisVectors, 0)
        #print(intBasisVectors)
        Dim = len(intBasisVectors)
        print("Hilbert space dim = " + str(Dim))

        if (nbrEigenstate > Dim):
            nbrEigenstate = Dim - 2
            
        links = GenLatticeNNLinksOptimized(L)
            
        if (c4Flag == False):
        
            if (loadHamiltonian == False):
                HOneBody = sp.sparse.lil_matrix((Dim,Dim), dtype=complex)

                #linksVer = GenLatticeNNLinks(L)[0]
                #linksHor = GenLatticeNNLinks(L)[1]

                HOneBody = BuildHOneBodyOptimized(basisVectors, intBasisVectors, links, J, FluxDensity, trapConf, gamma=gamma)
                # Convert it to a CSR sparse matrix
                HOneBody = HOneBody.tocsr()
                
                # Save the sparse Hamiltonian into a file
                if (saveHamiltonian == True):
                    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, N=N, alpha=FluxDensity)
                    gm.SaveMatrix(fileName, HOneBody)
                    
            else:
                fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, N=N, alpha=FluxDensity)
                HOneBody = gm.LoadMatrix(fileName)
            
            E, eVec = diagH(HOneBody, nbrEigenstate)

            print("Energy spectrum:")
            sortIdx = np.argsort(E)
            E = E[sortIdx]
            eVec = eVec[:,sortIdx]
            print(E)
            
            # Save the energy spectrum
            if (saveSpectrum == True):
                fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, 0, spectrum=True, alpha=FluxDensity, N=N)
                SaveSpectrum(fileName, E)

            # Save the eigenvectors
            if (saveEigenstates == True):
                for nEigenstate in np.arange(0,nbrEigenstate):
                    fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, alpha=FluxDensity, N=N)
                    SaveVector(fileName, eVec[:,nEigenstate])

        ######## C4 HARDCORE ########
        else:
            print('Using c4 symmetry')
            print('Generating C4 basis...')
            hardcoreC4Reps = GenerateHardcoreC4Basis(basisVectors, L)
            #print(hardcoreC4Reps)
            redDim = len(hardcoreC4Reps)
            print(f'Reduced Hilbert space dimension = {redDim}')
            HOneBodyC4 = sp.sparse.lil_matrix((redDim,redDim), dtype=complex)

            #binaryStateStrings = [IntToBinary(stateArray,Ns) for stateArray in intBasisVectors]
            #print('--- ORDER OF THE BASIS VECTORS ---')
            #print(f'states={binaryStateStrings}')

            for c4Sector in np.arange(0,4):
                start = time.time()
                HOneBodyC4 = BuildHOneBodyC4SymmetryOptimized(hardcoreC4Reps, basisVectors, L, links, J, FluxDensity, trapConf, gamma, c4Sector)
                TimePrint(start)

                #print("HAMILTONIAN:")
                #print(HOneBodyC4)
                E, eVec = diagH(HOneBodyC4, nbrEigenstate)

                print(f'Energies for sector l={c4Sector}:')
                E.sort()
                print(E)
                
                if (saveSpectrum == True):
                    fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, 0, spectrum=True, alpha=FluxDensity, c4=True, N=N)
                    SaveC4Spectrum(fileName, c4Sector, E)
                    
                if (saveHamiltonian == True):
                    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, alpha=FluxDensity, c4=True, N=N)
                    gm.SaveMatrix(fileName, HOneBodyC4)

                
    else:
        print('-----------------------------------------')
        print('Softcore mode')
        print(f'Two-body onsite interaction U={U}')
        print(f'Three-body onsite interaction U3={U3}')
        print('-----------------------------------------')
        print('Generating basis vectors...')
        softcoreBasis = [ [int(bit) for bit in string] for string in GenerateBasis(N, Ns) ]
        Dim = len(softcoreBasis)
        print("Hilbert space (soft-core) dim = " + str(len(softcoreBasis)))
        basisTags = PomeranovTag(softcoreBasis)
        #print(softcoreBasis)
        #print(basisTags)
        links = GenLatticeNNLinksOptimized(L)
        
        if (c4Flag == False):
        
            if (loadHamiltonian == False):
                # Build the tight-binding Hamiltonian
                HOneBodySoftCore = BuildHOneBodyOptimizedSoftCore(softcoreBasis, basisTags, links, J, FluxDensity, trapConf, gamma=gamma)
                HOneBodySoftCore = HOneBodySoftCore.tocsr()
                H = HOneBodySoftCore
                
                # Add the two-body onsite interaction
                if (U != 0):
                    HTwoBodyOnsite = BuildHTwoBodyOnsite(softcoreBasis, basisTags, L, U)
                    HTwoBodyOnsite = HTwoBodyOnsite.tocsr()
                    H = H + HTwoBodyOnsite
                    
                # Add the three-body onsite interaction
                if (U3 != 0):
                    HThreeBodyOnsite = BuildHThreeBodyOnsite(softcoreBasis, basisTags, L, U3)
                    HThreeBodyOnsite = HThreeBodyOnsite.tocsr()
                    H = H + HThreeBodyOnsite
                    
                # Save the sparse Hamiltonian into a file
                if (saveHamiltonian == True):
                    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, U3=U3, alpha=FluxDensity, N=N)
                    gm.SaveMatrix(fileName, H)
            else:
                fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, U3=U3, alpha=FluxDensity, N=N)
                H = gm.LoadMatrix(fileName)

            if (nbrEigenstate > Dim):
                nbrEigenstate = Dim - 2
                
            E, eVec = diagH(H, nbrEigenstate)
            E.sort()
            print("Energy spectrum:")
            print(E)
            
            if (saveSpectrum == True):
                fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, 0, spectrum=True, U3=U3, alpha=FluxDensity, N=N)
                SaveSpectrum(fileName, E)
            
            if (saveEigenstates == True):
                for nEigenstate in np.arange(0,nbrEigenstate):
                    fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, U3=U3, alpha=FluxDensity, N=N)
                    SaveVector(fileName, eVec[:,nEigenstate])
                
        ######## C4 SOFTCORE ########
        else:
            softcoreC4Reps = GenerateSoftcoreC4Basis(softcoreBasis, L)
            #print(softcoreC4Reps)
            redDim = len(softcoreC4Reps)
            print(f'Reduced Hilbert space dimension = {redDim}')
            
            HOneBodySoftcoreC4 = sp.sparse.lil_matrix((redDim,redDim), dtype=complex)
            for c4Sector in np.arange(0,4):
                HOneBodySoftcoreC4 = BuildHOneBodySoftCoreC4SymmetryOptimized(softcoreC4Reps, softcoreBasis, L, links, J, FluxDensity, trapConf, gamma, c4Sector)
                HOneBodySoftcoreC4 = HOneBodySoftcoreC4.tocsr()
                H = HOneBodySoftcoreC4
                
                if (U != 0):
                    HTwoBodyOnsiteC4 = BuildHTwoBodyOnsiteC4Optimized(softcoreC4Reps, softcoreBasis, L, U)
                    HTwoBodyOnsiteC4 = HTwoBodyOnsiteC4.tocsr()
                    H = H + HTwoBodyOnsiteC4
                    
                if (U3 != 0):
                    HThreeBodyOnsiteC4 = BuildHThreeBodyOnsiteC4Optimized(softcoreC4Reps, softcoreBasis, L, U3)
                    HThreeBodyOnsiteC4 = HThreeBodyOnsiteC4.tocsr()
                    H = H + HThreeBodyOnsiteC4

                E, eVec = diagH(H, nbrEigenstate)

                print(f'Energies for sector l={c4Sector}:')
                E.sort()
                print(E)
                
                if (saveSpectrum == True):
                    fileName = GenFilename(hardcore, L, J, U, trapConf, gamma, 0, spectrum=True, alpha=FluxDensity, c4=True, U3=U3, N=N)
                    SaveC4Spectrum(fileName, c4Sector, E)
                    
                if (saveHamiltonian == True):
                    fileName = gm.GenFilename(hardcore, L, J, U, trapConf, gamma, 0, hamiltonian=True, alpha=FluxDensity, c4=True, U3=U3, N=N)
                    gm.SaveMatrix(fileName, H)
