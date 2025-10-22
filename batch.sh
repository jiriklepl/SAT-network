#!/usr/bin/env bash

#SBATCH --job-name=sat_network
#SBATCH --output=logs/sat_network_%j.out
#SBATCH --error=logs/sat_network_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=mpi-homo-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

if [ ! -f ./.venv/bin/activate ]; then
    echo "Cannot find virtual environment. Please create one first." >&2
    exit 1
fi

. ./.venv/bin/activate

./main.py "$@"
