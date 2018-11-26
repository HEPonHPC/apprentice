#!/bin/bash

# ./mkPlots.sh ../f1_test.txt plots_f1_s1_t100 ../scripts/JSON/f1_s1_*t100.json
#
# Arguments:
#     1. input file with test points
#     2. output directory for plots (will be created)
#     all others: json files


ftest=$1
OUTDIR=$2
mkdir -p $OUTDIR

for fin in ${@:3}
do
    btest=`basename $ftest`
    bin=`basename $fin`
    fout="$OUTDIR/${btest%.*}_${bin%.*}.png"
    python ../scripts/bm-plot2D.py $fin -t $ftest -n 1 -o $fout
done

cd $OUTDIR
for n in {0..4};do convert  *n$n*png +append row_n$n.png;done
convert row_n*png -append ${OUTDIR}.png
cd -
