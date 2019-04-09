#!/bin/bash

mkdir -p benchmarkdata
pycodepath="../benchmarkdata/scripts/mkBenchmarkData.py "

for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "benchmarkdata/f${i}.txt" ];               then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}.txt                         -n 1000   -r 0                        -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ];  then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt            -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ];  then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt            -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ];  then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt            -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ];  then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt            -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i}; fi done

for i in 7; do if [ ! -e "benchmarkdata/f${i}.txt" ];              then python $pycodepath -s 9876512 -o benchmarkdata/f${i}.txt                          -n 1000   -r 0                        -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt             -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt             -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt             -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt             -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin 0 --xmax 1; fi done

for i in 10; do if [ ! -e "benchmarkdata/f${i}.txt" ];                 then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}.txt                         -d 4 -n 1000   -r 0                         -f ${i}; fi done
for i in 10; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 10; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 10; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 10; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done

for i in 17; do if [ ! -e "benchmarkdata/f${i}.txt" ];              then python $pycodepath -s 9876512 -o benchmarkdata/f${i}.txt                          -d 3 -n 1000   -r 0                        -f ${i}; fi done
for i in 17; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i}; fi done
for i in 17; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i}; fi done
for i in 17; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i}; fi done
for i in 17; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i}; fi done

for i in 18; do if [ ! -e "benchmarkdata/f${i}.txt" ];              then python $pycodepath -s 9876512 -o benchmarkdata/f${i}.txt                          -d 4 -n 1000   -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ]; then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done

for i in 19; do if [ ! -e "benchmarkdata/f${i}.txt" ];                 then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}.txt                         -d 4 -n 1000   -r 0                         -f ${i}; fi done
for i in 19; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 19; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 19; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 19; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ];    then python $pycodepath -c -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done

for i in 20; do if [ ! -e "benchmarkdata/f${i}.txt" ];                 then python $pycodepath -s 9876512 -o benchmarkdata/f${i}.txt                         -d 7 -n 1000   -r 0                         -f ${i}; fi done
for i in 20; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt            -d 7 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 20; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt            -d 7 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 20; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt            -d 7 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 20; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt            -d 7 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done

for i in 21; do if [ ! -e "benchmarkdata/f${i}.txt" ];                 then python $pycodepath -s 9876512 -o benchmarkdata/f${i}.txt                         -d 2 -n 1000   -r 0                         -f ${i}; fi done
for i in 21; do if [ ! -e "benchmarkdata/f${i}_noisepct10-1.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-1.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 21; do if [ ! -e "benchmarkdata/f${i}_noisepct10-3.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-3.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 21; do if [ ! -e "benchmarkdata/f${i}_noisepct10-6.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-6.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 21; do if [ ! -e "benchmarkdata/f${i}_noisepct10-9.txt" ];    then python $pycodepath -s 9876512 -o benchmarkdata/f${i}_noisepct10-9.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done
