#!/bin/bash
# ./runrappsip.sh ../benchmarkdata/f12.txt f12 2 2 2x
# ./runrappsip.sh ../workflow/data/DM_6D.h5 6D 0 0 2x
# ./runrappsip.sh ../benchmarkdata/f21.txt f21_2x 2 2 2x
programname=$0
function usage {
    echo "usage: $programname infile fndesc mmax nmax ts"
    exit 1
}

if [ $# -lt 5 ]; then
    usage
fi

infile=$1;
fndesc=$2;
mmax=$3;
nmax=$4;
ts=$5;

mkdir -p $fndesc/out;
mkdir -p $fndesc/log/consolelog;
# python runrappsip.py ../benchmarkdata/f12.txt f12 2 2 1x 6d
for pdeg in $(seq 1 $mmax); do
    for qdeg in $(seq 1 $nmax); do
      if [[ ( "$pdeg" -eq 0 && "$qdeg" -eq 0 )]]
        then continue
      fi
      colsolelog=$fndesc"/log/consolelog/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".log";
      outfile=$fndesc"/out/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".json";
      if [ ! -f "$outfile" ]
      then
        echo $consolelog;
        nohup python runrappsip.py $infile $fndesc $pdeg $qdeg $ts $fndesc >$colsolelog 2>&1 &
      fi
      # python runrappsip.py $infile $fndesc $pdeg $qdeg $ts $fndesc >$colsolelog 2>&1
    done
done
