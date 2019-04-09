#!/bin/bash
# ./runpappforsimcoeffs.sh ../benchmarkdata/f12.txt f12 2 2 2x
# ./runpappforsimcoeffs.sh ../workflow/data/DM_6D.h5 6D 0 0 2x
# ./runpappforsimcoeffs.sh ../benchmarkdata/f21.txt f21_2x 2 2 2x
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

mkdir -p $fndesc/outpa;
mkdir -p $fndesc/log/consolelogpa;
# python runrappsip.py ../benchmarkdata/f12.txt f12 2 2 1x 6d
for pdeg in $(seq 0 $mmax); do
    for qdeg in $(seq 0 $nmax); do
      if [[ ( "$pdeg" -eq 0 && "$qdeg" -eq 0 )]]
        then continue
      fi
      colsolelog=$fndesc"/log/consolelogpa/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".log";
      outfile=$fndesc"/outpa/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".json";
      if [ ! -f "$outfile" ]
      then
        echo $consolelog;
        nohup python runpappforsimcoeffs.py $infile $fndesc $pdeg $qdeg $ts "$outfile" >$colsolelog 2>&1 &
      fi
    done
done
