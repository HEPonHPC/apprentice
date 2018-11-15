#!/bin/bash

declare -a testSize=(25 50 100 200 500 1000)

fin=$1

for t in "${testSize[@]}"
do
    parallel -j 6 --bar ./bm-mkapp.py  $fin -O JSON -m {1} -n {2} -s {3} -t $t ::: {0..5} ::: {0..5} ::: {1..3}
done

