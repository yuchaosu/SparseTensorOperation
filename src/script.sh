#!/bin/bash

# Loop through different thread configurations
for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads
    echo "Running with $threads threads"

    # Execute the tensor program
    ./hicoo

    # Measure power statistics
    #sudo powerstat -R ./coo

    # Collect performance statistics
    #perf stat -e cache-misses,cache-references ./coo
done
