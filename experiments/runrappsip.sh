#!/bin/bash
# ./runrappsip.sh ../benchmarkdata/f12.txt f12_2x 0 2 0 2 2x
# ./runrappsip.sh ../workflow/data/DM_6D.h5 6D_2x 0 2 0 2 2x
programname=$0
function usage {
    echo "usage: $programname infile fndesc mmin mmax nmin nmax ts"
    exit 1
}

if [ $# -lt 7 ]; then
    usage
fi

infile=$1;
fndesc=$2;
mmin=$3;
mmax=$4;
nmin=$5;
nmax=$6;
ts=$7;

mkdir -p $fndesc/out;
mkdir -p $fndesc/log/consolelog;
# python runrappsip.py ../benchmarkdata/f12.txt f12 2 2 1x 6d
for pdeg in $(seq $mmin $mmax); do
    for qdeg in $(seq $nmin $nmax); do
      colsolelog=$fndesc"/log/consolelog/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".log";
      echo $consolelog;
      nohup python runrappsip.py $infile $fndesc $pdeg $qdeg $ts $fndesc >$colsolelog 2>&1 &
      # python runrappsip.py $infile $fndesc $pdeg $qdeg $ts $fndesc >$colsolelog 2>&1
    done
done
