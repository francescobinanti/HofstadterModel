import numpy as np
import scipy as sp
import sys
import os
import argparse
import time
import math
    
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
    
def HilbertDim(N,Ns,hardcore):
    """
    Calculate the Hilbert space dimension for N particles in Ns sites
    for hardcore or softcore bosons
    """
    if hardcore == False:
        return sp.special.comb(N+Ns-1,N)
    else:
        return sp.special.comb(Ns,N)
    
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
    
def GenFilename(hardcore, L, J, U, trapConf, gamma, nEigenstate, hamiltonian=False, spectrum=False, absSpectrum=False, localDensity=False, c4=False, U3=0.0, alpha=0.0, N=0, r0=0.0, corrFunction=0, excFrac=False, timeEvolOmega=0.0, timeEvolAngMom=0, timeEvolEps=0.0, densityEvolution=False, evolvedState=False, timeEvolState=0.0):
    """
    Returns the filename string the saved eigenstates will have
    """
    tmpString = ""
    fileName = f'bosons'
    
    if (hardcore == True):
        tmpString = f'_hardcore'
        fileName = fileName + tmpString
    else:
        tmpString = f'_softcore'
        fileName = fileName + tmpString
        
    if (c4 == True):
        tmpString = f'_c4'
        fileName = fileName + tmpString
        
    tmpString = f'_N_{N}_x_{L}_y_{L}_J_{J}_alpha_{alpha}'
    fileName = fileName + tmpString
    
    if ((hardcore == False) and (U != 0.)):
        tmpString = f'_U_{U}'
        fileName = fileName + tmpString
    
    if ((hardcore == False) and (U3 != 0.)):
        tmpString = f'_U3_{U3}'
        fileName = fileName + tmpString
    
    if (trapConf != 0):
        tmpString = f'_c_{trapConf}_g_{gamma}'
        fileName = fileName + tmpString
        
    if (hamiltonian == False) and (spectrum == False) and (absSpectrum == False):
        tmpString = f'_n_{nEigenstate}'
        fileName = fileName + tmpString
        
    if (hamiltonian == True):
        tmpString = f'_hamiltonian'
        fileName = fileName + tmpString
        
    if (spectrum == True):
        tmpString = f'_spectrum'
        fileName = fileName + tmpString
        
    # TIME EVOLUTION / LASER PARAMETERS
    if (r0 != 0.0):
        tmpString = f'_r0_{r0}'
        fileName = fileName + tmpString
        
    if (timeEvolEps != 0.0):
        tmpString = f'_eps_{timeEvolEps}'
        fileName = fileName + tmpString
    if (timeEvolOmega != 0):
        tmpString = f'_w_{timeEvolOmega}'
        fileName = fileName + tmpString
    if (timeEvolAngMom != 0.0):
        tmpString = f'_l_{timeEvolAngMom}'
        fileName = fileName + tmpString
        
    if (absSpectrum == True):
        tmpString = f'_absorption'
        fileName = fileName + tmpString
        
    if (localDensity == True):
        tmpString = f'_density'
        fileName = fileName + tmpString
        
    if (excFrac == True):
        tmpString = f'_exfraction'
        fileName = fileName + tmpString
            
    if (densityEvolution == True):
        tmpString = f'_densityevolution'
        fileName = fileName + tmpString
        
    if (corrFunction != 0):
        tmpString = f'_corrFunc_{corrFunction}'
        fileName = fileName + tmpString
        
    if (evolvedState == True):
        tmpString = f'_evolved'
        fileName = fileName + tmpString
        
    if timeEvolState != 0.0:
        tmpString = f'_t_{timeEvolState}'
        fileName = fileName + tmpString
        
    return fileName
    
def SaveSpectrum(fileName, energies):
    fileName = fileName + '.dat'
    fileDesc = open(fileName,"a")
    print(f'Saving the spectrum in: {fileName}')
    
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
        
def SaveArraysTwoColFile(fileName, xArray, yArray):
    fileName = fileName + '.dat'
    
    dataStack = np.column_stack((xArray, yArray))
    
    with open(fileName,"w") as fileDesc:
        np.savetxt(fileDesc, dataStack)

def SaveVector(fileName, eigenstate):
    """
    Routine to save eigenstates in .npy format (binary files)
    """
    print(f'Saving the eigenstate in: {fileName}')
    np.save(fileName, eigenstate)
    
def SaveMatrix(fileName, matrix):
    """
    Save a sparse matrix into a file
    """
    sp.sparse.save_npz(fileName,matrix)
    
def SaveTwoColFile(fileName, x, y):
    """
    Generic routine to save data in the {x--y} format
    """
    fileName = fileName + '.dat'
    
    with open(fileName, "ab") as fileDesc:
        np.savetxt(fileDesc, np.array([x,y]).reshape(1,2), newline='\n')
    
def LoadVector(fileName):
    """
    Routine to load data of an eigenstate from a binary file
    """
    fileName = fileName + '.npy'
    vector = np.load(fileName)
    
    return vector
    
def LoadMatrix(fileName):
    """
    Load a sparse matrix from a file .npz
    """
    return sp.sparse.load_npz(fileName + '.npz')
    
def LoadSpectrum(fileName):
    """
    Load the energy spectrum
    """
    fileName = fileName + '.dat'
    return np.loadtxt(fileName)
    
def LoadC4Spectrum(fileName):
    """
    Load the C4 energy spectrum data
    """
    fileName = fileName + '.dat'
    data = np.loadtxt(fileName)
    return data[:,0], data[:,1]
    
def LoadDensity(fileName):
    """
    Load the local density data
    """
    fileName = fileName + '.dat'
    data = np.loadtxt(fileName)
    return data[:,0], data[:,1], data[:,2]
    
def LoadFileThree(fileName):
    """
    Load data from file containing three columns of data delimited by spaces
    """
    fileName = fileName + '.dat'
    data = np.loadtxt(fileName)
    return data[:,0], data[:,1], data[:,2]
    
def LoadFileTwo(fileName):
    """
    Load data from file containing two columns of data delimited by spaces
    """
    fileName = fileName + '.dat'
    data = np.loadtxt(fileName)
    return data[:,0], data[:,1]
