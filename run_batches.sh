#!/usr/bin/env bash

set -euo pipefail

instructions=(14 15 16)
batch_sizes=(8 16)

encode_boolean_flags=("--encode-boolean" "")
force_ordered_flags=("--force-ordered" "")
force_useful_flags=("--force-useful" "")

shuffle_flags=("--no-shuffle" "--seed 42")

do_all_flags=("--do-all" "")

num_batches=$(( ${#instructions[@]} * ${#batch_sizes[@]} * ${#encode_boolean_flags[@]} * ${#force_ordered_flags[@]} * ${#force_useful_flags[@]} * ${#shuffle_flags[@]} * ${#do_all_flags[@]} ))

echo "Will create $num_batches jobs"

srun --ntasks=1 --cpus-per-task=2 --mem=2G --time=1:00 --partition mpi-homo-short bash setup.sh

for instr in "${instructions[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for encode_boolean in "${encode_boolean_flags[@]}"; do
            for force_ordered in "${force_ordered_flags[@]}"; do
                for force_useful in "${force_useful_flags[@]}"; do
                    for shuffle in "${shuffle_flags[@]}"; do
                        for do_all in "${do_all_flags[@]}"; do
                            sbatch batch.sh --instructions "$instr" --batch-size "$batch_size" $encode_boolean $force_ordered $force_useful $shuffle $do_all
                        done
                    done
                done
            done
        done
    done
done
