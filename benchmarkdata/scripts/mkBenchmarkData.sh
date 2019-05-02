#!/bin/bash

for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}.txt" ];               then python mkBenchmarkData.py -c -o ../f${i}.txt                         -n 1000   -r 0                        -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}_noisepct10-1.txt" ];  then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-1.txt            -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}_noisepct10-3.txt" ];  then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-3.txt            -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}_noisepct10-6.txt" ];  then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-6.txt            -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}_noisepct10-9.txt" ];  then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-9.txt            -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i}; fi done
for i in {1..6} 8 9 {12..16} 22; do if [ ! -e "../f${i}_test.txt" ];          then python mkBenchmarkData.py -c -o ../f${i}_test.txt          -s 9999   -n 100000 -r 0                        -f ${i}; fi done
# for i in {1..6} 8 9; do if [ ! -e "../f${i}_noise_0.1_test.txt" ]; then python mkBenchmarkData.py -c -o ../f${i}_noise_0.1_test.txt -s 9999 -n 100000 -r 0.1 -f ${i}; fi done
# for i in {1..6} 8 9; do if [ ! -e "../f${i}_noise_0.5_test.txt" ]; then python mkBenchmarkData.py -c -o ../f${i}_noise_0.5_test.txt -s 9999 -n 100000 -r 0.5 -f ${i}; fi done

for i in 7; do if [ ! -e "../f${i}.txt" ];              then python mkBenchmarkData.py -o ../f${i}.txt                          -n 1000   -r 0                        -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "../f${i}_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt             -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "../f${i}_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt             -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "../f${i}_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt             -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "../f${i}_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt             -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin 0 --xmax 1; fi done
for i in 7; do if [ ! -e "../f${i}_test.txt" ];         then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999   -n 100000 -r 0                        -f ${i} --xmin 0 --xmax 1; fi done
# for i in 7; do if [ ! -e "../f${i}_noise_0.1_test.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noise_0.1_test.txt -s 9999 -n 100000 -r 0.1 -f ${i} --xmin 0 --xmax 1; fi done
# for i in 7; do if [ ! -e "../f${i}_noise_0.5_test.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noise_0.5_test.txt -s 9999 -n 100000 -r 0.5 -f ${i} --xmin 0 --xmax 1; fi done

for i in 10; do if [ ! -e "../f${i}.txt" ];                 then python mkBenchmarkData.py -c -o ../f${i}.txt                         -d 4 -n 1000   -r 0                         -f ${i}; fi done
for i in 10; do if [ ! -e "../f${i}_noisepct10-1.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-1.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 10; do if [ ! -e "../f${i}_noisepct10-3.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-3.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 10; do if [ ! -e "../f${i}_noisepct10-6.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-6.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 10; do if [ ! -e "../f${i}_noisepct10-9.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-9.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done
for i in 10; do if [ ! -e "../f${i}_test.txt" ];            then python mkBenchmarkData.py -c -o ../f${i}_test.txt           -s 9999  -d 4 -n 100000 -r 0                         -f ${i}; fi done
# for i in 10; do if [ ! -e "../f${i}_noise_0.1_test.txt" ]; then python mkBenchmarkData.py -c -o ../f${i}_noise_0.1_test.txt -s 9999 -d 4 -n 100000 -r 0.1 -f ${i}; fi done
# for i in 10; do if [ ! -e "../f${i}_noise_0.5_test.txt" ]; then python mkBenchmarkData.py -c -o ../f${i}_noise_0.5_test.txt -s 9999 -d 4 -n 100000 -r 0.5 -f ${i}; fi done

for i in 17; do if [ ! -e "../f${i}.txt" ];              then python mkBenchmarkData.py -o ../f${i}.txt                          -d 3 -n 1000   -r 0                        -f ${i}; fi done
for i in 17; do if [ ! -e "../f${i}_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i}; fi done
for i in 17; do if [ ! -e "../f${i}_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i}; fi done
for i in 17; do if [ ! -e "../f${i}_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i}; fi done
for i in 17; do if [ ! -e "../f${i}_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i}; fi done
for i in 17; do if [ ! -e "../f${i}_test.txt" ];         then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999   -d 3 -n 100000 -r 0                        -f ${i}; fi done

for i in 18; do if [ ! -e "../f${i}.txt" ];              then python mkBenchmarkData.py -o ../f${i}.txt                          -d 4 -n 1000   -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "../f${i}_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "../f${i}_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "../f${i}_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "../f${i}_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt             -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 18; do if [ ! -e "../f${i}_test.txt" ];         then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999   -d 4 -n 100000 -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done

for i in 23; do if [ ! -e "../f${i}.txt" ];              then python mkBenchmarkData.py -o ../f${i}.txt                          -d 3 -n 1000   -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 23; do if [ ! -e "../f${i}_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 23; do if [ ! -e "../f${i}_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 23; do if [ ! -e "../f${i}_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 23; do if [ ! -e "../f${i}_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 23; do if [ ! -e "../f${i}_test.txt" ];         then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999   -d 3 -n 100000 -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done

# f23-1 is f23 with box [-0.999999, 0.999999]
for i in 23; do if [ ! -e "../f${i}-1.txt" ];              then python mkBenchmarkData.py -o "../f${i}-1.txt"                          -d 3 -n 1000   -r 0                        -f ${i} --xmin -0.999999 --xmax 0.999999; fi done
for i in 23; do if [ ! -e "../f${i}-1_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o "../f${i}-1_noisepct10-1.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.999999 --xmax 0.999999; fi done
for i in 23; do if [ ! -e "../f${i}-1_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o "../f${i}-1_noisepct10-3.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.999999 --xmax 0.999999; fi done
for i in 23; do if [ ! -e "../f${i}-1_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o "../f${i}-1_noisepct10-6.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.999999 --xmax 0.999999; fi done
for i in 23; do if [ ! -e "../f${i}-1_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o "../f${i}-1_noisepct10-9.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.999999 --xmax 0.999999; fi done
for i in 23; do if [ ! -e "../f${i}-1_test.txt" ];         then python mkBenchmarkData.py -o "../f${i}-1_test.txt"           -s 9999   -d 3 -n 100000 -r 0                        -f ${i} --xmin -0.999999 --xmax 0.999999; fi done

# f23-2 is f23 with box (-0.9, 0.9)
for i in 23; do if [ ! -e "../f${i}-2.txt" ];              then python mkBenchmarkData.py -o "../f${i}-2.txt"                          -d 3 -n 1000   -r 0                        -f ${i} --xmin -0.9 --xmax 0.9; fi done
for i in 23; do if [ ! -e "../f${i}-2_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o "../f${i}-2_noisepct10-1.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.9 --xmax 0.9; fi done
for i in 23; do if [ ! -e "../f${i}-2_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o "../f${i}-2_noisepct10-3.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.9 --xmax 0.9; fi done
for i in 23; do if [ ! -e "../f${i}-2_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o "../f${i}-2_noisepct10-6.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.9 --xmax 0.9; fi done
for i in 23; do if [ ! -e "../f${i}-2_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o "../f${i}-2_noisepct10-9.txt"             -d 3 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.9 --xmax 0.9; fi done
for i in 23; do if [ ! -e "../f${i}-2_test.txt" ];         then python mkBenchmarkData.py -o "../f${i}-2_test.txt"           -s 9999   -d 3 -n 100000 -r 0                        -f ${i} --xmin -0.9 --xmax 0.9; fi done

for i in 24; do if [ ! -e "../f${i}.txt" ];              then python mkBenchmarkData.py -o ../f${i}.txt                          -d 2 -n 1000   -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 24; do if [ ! -e "../f${i}_noisepct10-1.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt             -d 2 -n 1000   -r $(bc -l <<< "10 ^(-1)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 24; do if [ ! -e "../f${i}_noisepct10-3.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt             -d 2 -n 1000   -r $(bc -l <<< "10 ^(-3)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 24; do if [ ! -e "../f${i}_noisepct10-6.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt             -d 2 -n 1000   -r $(bc -l <<< "10 ^(-6)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 24; do if [ ! -e "../f${i}_noisepct10-9.txt" ]; then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt             -d 2 -n 1000   -r $(bc -l <<< "10 ^(-9)")  -f ${i} --xmin -0.95 --xmax 0.95; fi done
for i in 24; do if [ ! -e "../f${i}_test.txt" ];         then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999   -d 2 -n 100000 -r 0                        -f ${i} --xmin -0.95 --xmax 0.95; fi done

for i in 19; do if [ ! -e "../f${i}.txt" ];                 then python mkBenchmarkData.py -c -o ../f${i}.txt                         -d 4 -n 1000   -r 0                         -f ${i}; fi done
for i in 19; do if [ ! -e "../f${i}_noisepct10-1.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-1.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 19; do if [ ! -e "../f${i}_noisepct10-3.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-3.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 19; do if [ ! -e "../f${i}_noisepct10-6.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-6.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 19; do if [ ! -e "../f${i}_noisepct10-9.txt" ];    then python mkBenchmarkData.py -c -o ../f${i}_noisepct10-9.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done
for i in 19; do if [ ! -e "../f${i}_test.txt" ];            then python mkBenchmarkData.py -c -o ../f${i}_test.txt           -s 9999  -d 4 -n 100000 -r 0                         -f ${i}; fi done

for i in 20; do if [ ! -e "../f${i}.txt" ];                 then python mkBenchmarkData.py -o ../f${i}.txt                         -d 4 -n 1000   -r 0                         -f ${i}; fi done
for i in 20; do if [ ! -e "../f${i}_noisepct10-1.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 20; do if [ ! -e "../f${i}_noisepct10-3.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 20; do if [ ! -e "../f${i}_noisepct10-6.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 20; do if [ ! -e "../f${i}_noisepct10-9.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt            -d 4 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done
for i in 20; do if [ ! -e "../f${i}_test.txt" ];            then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999  -d 4 -n 100000 -r 0                         -f ${i}; fi done

for i in 21; do if [ ! -e "../f${i}.txt" ];                 then python mkBenchmarkData.py -o ../f${i}.txt                         -d 2 -n 1000   -r 0                         -f ${i}; fi done
for i in 21; do if [ ! -e "../f${i}_noisepct10-1.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-1.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-1)")   -f ${i}; fi done
for i in 21; do if [ ! -e "../f${i}_noisepct10-3.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-3.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-3)")   -f ${i}; fi done
for i in 21; do if [ ! -e "../f${i}_noisepct10-6.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-6.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-6)")   -f ${i}; fi done
for i in 21; do if [ ! -e "../f${i}_noisepct10-9.txt" ];    then python mkBenchmarkData.py -o ../f${i}_noisepct10-9.txt            -d 2 -n 1000   -r $(bc -l <<< "10 ^(-9)")   -f ${i}; fi done
for i in 21; do if [ ! -e "../f${i}_test.txt" ];            then python mkBenchmarkData.py -o ../f${i}_test.txt           -s 9999  -d 2 -n 100000 -r 0                         -f ${i}; fi done
