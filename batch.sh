#!/usr/bin/env bash

#SBATCH --job-name=sat_network
#SBATCH --output=logs/sat_network_%j.out
#SBATCH --error=logs/sat_network_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=mpi-homo-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail

BIN="${SAT_SYNTH_CPP:-}"
if [ -z "$BIN" ]; then
    for candidate in \
        ./build/release/sat_synth_cpp \
        ./build/release-with-debug-info/sat_synth_cpp \
        ./build/debug/sat_synth_cpp \
        ./build/sat_synth_cpp; do
        if [ -x "$candidate" ]; then
            BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "Cannot find sat_synth_cpp binary. Build it first with: cmake --preset release && cmake --build --preset release -j" >&2
    exit 1
fi

exec "$BIN" "$@"
