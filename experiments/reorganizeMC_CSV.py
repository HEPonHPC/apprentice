import sys,os
import argparse
import numpy as np
import warnings
import functools
import csv

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

@deprecated
def prepareMCdata(args):
    import pandas as pd
    allMC =[]
    X = np.array([])
    for fno,file in enumerate(args.DATAFILES):
        data = pd.read_csv(file, header=None)
        D = data.values
        MC = D[:, -3]
        if fno == 0:
            X = np.array(D[:, :-3])
        allMC.append(MC)

    avgMC = []
    sdMC = []
    for j in range(len(X)):
        MCperP = [allMC[i][j] for i in range(len(allMC))]
        avgMC.append(np.mean(MCperP))
        sdMC.append(np.std(MCperP))
    avgMC = np.atleast_2d(avgMC)
    sdMC = np.atleast_2d(sdMC)
    # print(X)
    # print(avgMC)
    # print(avgDMC)
    dir = os.path.dirname(args.OUTFILE)
    os.makedirs(dir,exist_ok=True)
    np.savetxt(args.OUTFILE, np.hstack((X, avgMC.T,sdMC.T)), delimiter=",")

def prepareFlatMCdata(args):
    import pandas as pd
    allX = np.array([])
    allMC = np.array([])
    allDeltaMC = np.array([])
    for fno, file in enumerate(args.DATAFILES):
        data = pd.read_csv(file, header=None)
        D = data.values
        nparam = np.shape(D)[1] - 9
        MC = D[:, nparam]
        DeltaMC = D[:, nparam+1]
        X = np.array(D[:, :nparam])
        if fno ==0:
            allX = X
            allMC = MC
            allDeltaMC = DeltaMC
        else:
            allX = np.append(allX,X,axis=0)
            allMC = np.append(allMC,MC,axis=0)
            allDeltaMC = np.append(allDeltaMC, DeltaMC, axis=0)
        if args.NUMBER > 0 and fno  == args.NUMBER - 1 :
            break

    allMC = np.atleast_2d(allMC)
    allDeltaMC = np.atleast_2d(allDeltaMC)
    # print(X)
    # print(avgMC)
    # print(avgDMC)
    dir = os.path.dirname(args.OUTFILE)
    os.makedirs(dir, exist_ok=True)
    np.savetxt(args.OUTFILE, np.hstack((allX, allMC.T, allDeltaMC.T)), delimiter=",")

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Data for GP run',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-o", "--outtfile", dest="OUTFILE", type=str, default=None,
                        help="Output File")
    parser.add_argument("-i", "--inputcsvfiles", dest="DATAFILES", type=str, default=[], nargs='+',
                        help="Input MC File(s).")
    parser.add_argument("-t", "--type", dest="TYPE", type=str, default="MCDMCTime_FlatMCDMC",
                               choices=["MCDMCTime_AvgMCDMC", "MCDMCTime_FlatMCDMC"], help="Type")
    parser.add_argument("-n", "--number", dest="NUMBER", type=int, default=-1,
                        help="Number of samples to be flattened in \"MCDMCTime_FlatMCDMC\" type.\n"
                             "If n > 0 is specified, the first n samples will be flattened")

    args = parser.parse_args()
    if args.TYPE == "MCDMCTime_AvgMCDMC":
        print("Please use Code/ParameterTuning-3D/combineSamples.py in parameter-tuning-MC project.")
        print("This functionality has been deprecated")
        sys.exit(0)
        # prepareMCdata(args)
    elif args.TYPE == "MCDMCTime_FlatMCDMC":
        prepareFlatMCdata(args)

