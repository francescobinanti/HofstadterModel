#!/bin/bash

# Run the full time-dependent protocol for different frequencies w

# USAGE: ./RunTimeEvolutionProtocol.sh [N] [L] [conf] [gamma] [alpha] [r0] [omega init] [omega final] [angular momentum]

# Run the time evolution with ED for different frequencies, to generate excitation fraction and
# most important the density values at all times
for w in $(seq "$7" 0.01 "$8"); do
        python TimeEvolutionLaguerreGauss.py -N "$1" -L "$2" --conf "$3" --gamma "$4" --alpha "$5" --hardcore -r0 "$6" --angmom "$9" --omega $w --epsilon 0.05 --dt 0.1 --tmax 51.0 --density --squares 2
done
