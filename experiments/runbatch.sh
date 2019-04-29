#!/bin/bash


if [[ ! -f $1 ]]; then
	echo "usage: runbatch.sh <cmdlist_file> [<time_limit>]"
	exit 1
else
	cmdlist=$1
	timelimit=$2
fi

# nodelist="0"

declare -i nProcessor
declare -i nRunning
declare -i i
declare -i nextprob
declare -a processor
declare -a procid

nProcessor=0
for i in {1..192}; do
	processor[$nProcessor]=$i
	procid[$nProcessor]=0
	let ++nProcessor
done

function checkRunningProcesses ()
{
    nRunning=0
    for (( i=0 ; i < nProcessor ; ++i )); do
	if [[ "${procid[i]}" != "0" ]]; then
	    if [[ ! `ps h ${procid[i]}` ]]; then
		procid[$i]=0
	    else
		let ++nRunning
	    fi
	fi
    done
}

function findFreeProc ()
{
    started="no"
    for (( i=0 ; i < nProcessor ; ++i )); do
	if (( procid[$i] == 0 )); then
            echo Running $prob on ${processor[i]} at time
            date
	   curpwd=$PWD  taskset -c $i $prob
	    procid[$i]=$!
	    started="yes"
	    return
	fi
    done
}

nextprob=1

while (( $nextprob <= `grep -v '^#' $cmdlist | wc -l` )); do
    prob=`grep -v '^#' $cmdlist | sed -e "$nextprob q" | tail -1`
    checkRunningProcesses
    findFreeProc
    if [[ $started == "no" ]]; then
	sleep 10
    else
       let ++nextprob
    fi
done

# nRunning=1
#
# while (( $nRunning != 0 )); do
#     sleep 2
#     checkRunningProcesses
# done
