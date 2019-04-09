#!/bin/bash
# ./runpappforsimcoeffs.sh ../benchmarkdata/f12.txt f12 2 2 2x
# ./runpappforsimcoeffs.sh ../workflow/data/DM_6D.h5 6D 0 0 2x
# ./runpappforsimcoeffs.sh ../benchmarkdata/f21.txt f21_2x 2 2 2x
# fno=f7; noise=""; folder=$fno$noise"_2x"; testfile="../benchmarkdata/"$fno$noise".txt"; ./runnonsiprapp.sh $testfile $fno$noise"_2x" 8 8 2x
# fno=f15; noise="_noisepct10-1"; folder=$fno$noise"_2x"; testfile="../benchmarkdata/"$fno$noise".txt"; ./runnonsiprapp.sh $testfile $fno$noise"_2x" 8 8 2x
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

mkdir -p $fndesc/outra;
mkdir -p $fndesc/log/consolelogra;
# python runrappsip.py ../benchmarkdata/f12.txt f12 2 2 1x 6d
for pdeg in $(seq 1 $mmax); do
    for qdeg in $(seq 1 $nmax); do
      if [[ ( "$pdeg" -eq 0 && "$qdeg" -eq 0 )]]
        then continue
      fi
      colsolelog=$fndesc"/log/consolelogra/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".log";
      outfile=$fndesc"/outra/"$fndesc"_p"$pdeg"_q"$qdeg"_ts"$ts".json";
      if [ ! -f "$outfile" ]
      then
        echo $consolelog;
        nohup python runnonsiprapp.py $infile $fndesc $pdeg $qdeg $ts "$outfile" >$colsolelog 2>&1 &
      fi
    done
done
