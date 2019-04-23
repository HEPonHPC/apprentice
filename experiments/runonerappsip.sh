#!/bin/bash
# ./runrappsip.sh ../benchmarkdata/f12.txt f12 2 2 2x
# ./runrappsip.sh ../workflow/data/DM_6D.h5 6D 0 0 2x
# ./runrappsip.sh ../benchmarkdata/f21.txt f21_2x 2 2 2x
programname=$0
function usage {
    echo "usage: $programname infile fndesc m n ts"
    exit 1
}

if [ $# -lt 5 ]; then
    usage
fi

infile=$1;
fndesc=$2;
m=$3;
n=$4;
ts=$5;

mkdir -p $fndesc/out;
mkdir -p $fndesc/log/consolelog;
# python runrappsip.py ../benchmarkdata/f12.txt f12 2 2 1x 6d
if [[ ( "$m" -eq 0 && "$n" -eq 0 )]]
  then exit 1
fi
colsolelog=$fndesc"/log/consolelog/"$fndesc"_p"$m"_q"$n"_ts"$ts".log";
outfile=$fndesc"/out/"$fndesc"_p"$m"_q"$n"_ts"$ts".json";
if [ ! -f "$outfile" ]
  then
    echo $consolelog;
    nohup python runrappsip.py $infile $fndesc $m $n $ts $fndesc "$outfile" >$colsolelog 2>&1 &
fi
# python runrappsip.py $infile $fndesc $m $n $ts $fndesc >$colsolelog 2>&1
