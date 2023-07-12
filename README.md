# Topology in the Hofstadter Model
> Hofstadter Model on a square lattice loaded with interacting bosons.

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Usage](#usage)
* [Room for Improvement](#room-for-improvement)
* [Publications](#publications)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
This project is related to my research activity as a PhD student in the LPMMC laboratory, CNRS (Grenoble).

It provides an exact diagonalization of the Hofstadter model for interacting bosons on a square lattice:
$$H = -J\sum_{m,n} \bigg ( b^\dagger_{m+1,n} b_{m,n}e^{i2\pi\alpha n} + b^\dagger_{m,n+1} b_{m,n} + \text{h.c.} \bigg ) + \frac{U}{2}\sum_{m,n} b^\dagger_{m,n} b_{m,n} \left( b^\dagger_{m,n} b_{m,n} - 1 \right) + \sum_{m,n} V(m,n) b^\dagger_{m,n} b_{m,n}$$
It also gives a practical way to perform edge state spectroscopy via light-matter interaction, by mean of a
Laguerre-Gauss beam (periodic perturbation of the Hamiltonian above). The spatial modes of the LG laser are:
$$f_{n,l}(r,\theta) = L_n^l(r)\big( \frac{r}{r_0} \big)^l e^{\frac{r^2}{2r_0^2}} e^{i\theta l}$$
This gives a way to study the rich topological features of this model, that has already
been implemented in experiments and can be useful for future applications in quantum computing.

## Features
- Exact diagonalization of the Hofstadter Model with open boundary conditions
- Energy spectrum resolved in the C4 rotation symmetry
- Absorption spectra calculated via transition matrix elements (from the ground state) $\bra{\psi_n}f_{n,l}\ket{\psi_0}$
- Local particle density of a generic eigenstate of the Hamiltonian: $\bra{\psi_n}\hat n_i\ket{\psi_n}$

## Usage
```
usage: HofstadterThreeBody.py [-h] [-N N] [-L L] [-J J] [-U U] [-U3 U3] [--conf CONF] [--alpha ALPHA] [--hardcore [HARDCORE]]
                              [--savestates [SAVESTATES]] [--nbreigenstates NBREIGENSTATES] [--c4symmetry [C4SYMMETRY]]

options:
  -h, --help            show this help message and exit
  -N                   number of particles
  -L                   side of the square lattice of size LxL
  -J                   tunneling energy
  -U                   two-body onsite interaction (only in softcore mode)
  -U3                  three-body onsite interaction (only in softcore mode)
  --conf CONF           harmonic trap confinement strength (v0) as v0 * r^2
  --alpha ALPHA         magnetic flux density as alpha=p/q
  --hardcore [HARDCORE]
                        hardcore bosons mode
  --savestates [SAVESTATES]
                        save eigenvectors
  --nbreigenstates NBREIGENSTATES
                        number of eigenstates to be saved
  --c4symmetry [C4SYMMETRY]
                        use the c4 rotation symmetry in the exact diagonalization
```
```
usage: LaguerreGaussAbsorption.py [-h] [-N N] [-L L] [-J J] [-U U] [-U3 U3] [-r0 R0] [-n N] [--conf CONF] [--alpha ALPHA]
                                  [--hardcore [HARDCORE]] [--nbreigenstates NBREIGENSTATES]

options:
  -h, --help            show this help message and exit
  -N                   number of particles
  -L                   side of the square lattice of size LxL
  -J                   tunneling energy
  -U                   two-body onsite interaction (only in softcore mode)
  -U3                  three-body onsite interaction (only in softcore mode)
  -r0                  Laguerre-Gauss radius parameter (for a pure gaussian beam it is the gaussian dispersion)
  -n                   Radial order of the Laguerre polynomials used in the spatial mode of the LG beam
  --conf CONF           harmonic trap confinement strength (v0) as v0 * r^2
  --alpha ALPHA         magnetic flux density as alpha=p/q
  --hardcore [HARDCORE]
                        hardcore bosons mode
  --nbreigenstates NBREIGENSTATES
                        number of eigenstates to be considered in the matrix elements
```
```
usage: LocalDensity.py [-h] [-N N] [-L L] [-J J] [-U U] [-U3 U3] [--conf CONF] [--alpha ALPHA] [--hardcore [HARDCORE]]
                       [--neigenstate NEIGENSTATE]

options:
  -h, --help            show this help message and exit
  -N                   number of particles
  -L                   side of the square lattice of size LxL
  -J                   tunneling energy
  -U                   two-body onsite interaction (only in softcore mode)
  -U3                  three-body onsite interaction (only in softcore mode)
  --conf CONF           harmonic trap confinement strength (v0) as v0 * r^2
  --alpha ALPHA         magnetic flux density as alpha=p/q
  --hardcore [HARDCORE]
                        hardcore bosons mode
  --neigenstate NEIGENSTATE
                        index of the eigenstate for which the local density has to be calculated (e.g. 0 is the groundstate)
```

## Room for Improvement
Here some inspiration for improvement/features that might be done in the future, and an imminent to-do list.

Room for improvement/new features:
- Optimization for the building of the C4 symmetric Hamiltonian
- Add periodic boundary conditions (cylinder and torus geometries)

To be added soon:
- Time-dependent protocol to drive the system in the ground state with the LG perturbation and perform edge state spectroscopy

## Publications
[F. Binanti, N. Goldman, C. Repellin, ArXiv:2306.01624](https://arxiv.org/abs/2306.01624)
In this paper I used the code to benchmark an experimental protocol to detect topological edge states.

## Acknowledgements
Many features of this project were inspired by Cecile Repellin (LPMMC Grenoble, CNRS).

Many thanks to Nathan Goldman (Université libre de Bruxelles) for fruitful discussions.


## Contact
Francesco Binanti (francesco.binanti@lpmmc.cnrs.fr) - feel free to contact me!
