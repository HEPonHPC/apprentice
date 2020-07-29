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

def distToPoints(sumw, sumw2, xwidth):
    area = sumw
    height = area / xwidth
    areaErr = np.sqrt(sumw2)
    heightErr = areaErr / xwidth

    ret = np.empty((len(area), 2))
    ret[:, 0] = height
    ret[:, 1] = heightErr
    return height, heightErr

def add(data):
    osumw = 0.
    osumw2 = 0.
    SF1 = 0.
    xwidth = 0

    for dno, DD in enumerate(data):
        AA = DD.values
        nparam = np.shape(AA)[1] - 9
        SF1 = AA[:, -1 - 1]
        xwidth = AA[:, -2 - 1]
        sumw_1 = AA[:, nparam + 2]
        sumw_1 *= 1. / SF1
        osumw += sumw_1

        sumw2_1 = AA[:, nparam + 3]
        sumw2_1 *= 1. / SF1
        osumw2 += sumw2_1
        if args.NUMBER > 0 and dno == args.NUMBER - 1:
            break
    divi = len(data)
    if args.NUMBER > 0:
        divi = args.NUMBER
    osumw *= SF1 / divi
    osumw2 *= SF1 / divi

    return osumw, osumw2, xwidth

def createFlatAveragedData(args):
    import pandas as pd
    alldatapd = []
    X = None
    for fno,file in enumerate(args.DATAFILES):
        alldatapd.append(pd.read_csv(file, header=None))
        if fno == 0:
            AA = alldatapd[0].values
            nparam = np.shape(AA)[1] - 9
            X = np.array(AA[:, :nparam])
    if args.NUMBER <= 0:
        raise Exception("Number of samples per average is required and it has to be > 0")
    if args.NUMBER>len(alldatapd):
        raise Exception("number of samples to be combined cannot be greater than total number of samples")
    totalno = len(alldatapd)
    noperrun = args.NUMBER
    allX = np.array([])
    allMC = np.array([])
    allDeltaMC = np.array([])
    for i in range(0,totalno,noperrun):
        (height, heightErr) = distToPoints(*add(alldatapd[i:i+noperrun]))
        if i==0:
            allX = X
            allMC = height
            allDeltaMC = heightErr
        else:
            allX = np.append(allX, X, axis=0)
            allMC = np.append(allMC, height, axis=0)
            allDeltaMC = np.append(allDeltaMC, heightErr, axis=0)
    allMC = np.atleast_2d(allMC)
    allDeltaMC = np.atleast_2d(allDeltaMC)
    dir = os.path.dirname(args.OUTFILE)
    os.makedirs(dir, exist_ok=True)
    np.savetxt(args.OUTFILE, np.hstack((allX, allMC.T, allDeltaMC.T)), delimiter=",")

def createIndividualFlatAveragedData(args):
    import pandas as pd
    alldatapd = []
    X = None
    dir = os.path.dirname(args.OUTFILE)
    ofile = os.path.basename(args.OUTFILE)
    ofile = ofile.split(".")[0]
    os.makedirs(dir, exist_ok=True)
    for fno,file in enumerate(args.DATAFILES):
        alldatapd.append(pd.read_csv(file, header=None))
        if fno == 0:
            AA = alldatapd[0].values
            nparam = np.shape(AA)[1] - 9
            X = np.array(AA[:, :nparam])
    if args.NUMBER <= 0:
        raise Exception("Number of samples per average is required and it has to be > 0")
    if args.NUMBER>len(alldatapd):
        raise Exception("number of samples to be combined cannot be greater than total number of samples")
    totalno = len(alldatapd)
    noperrun = args.NUMBER
    index = 0
    for i in range(0,totalno,noperrun):
        (height, heightErr) = distToPoints(*add(alldatapd[i:i+noperrun]))
        allMC = np.atleast_2d(height)
        allDeltaMC = np.atleast_2d(heightErr)
        np.savetxt(os.path.join(dir,"{}_i{}.csv".format(ofile,index)),
                   np.hstack((X, allMC.T, allDeltaMC.T)), delimiter=",")
        index += 1

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
    parser.add_argument("-t", "--type", dest="TYPE", type=str, default="MCDMCww2wxwx2Time_FlatMCDMC",
                               choices=["MCDMCTime_AvgMCDMC",
                                        "MCDMCww2wxwx2Time_FlatMCDMC",
                                        "MCDMCww2wxwx2Time_IndvlFlatAvgMCDMC",
                                        "MCDMCww2wxwx2Time_FlatAvgMCDMC"], help="Type")
    parser.add_argument("-n", "--number", dest="NUMBER", type=int, default=-1,
                        help="Number of samples to be flattened in \"MCDMCww2wxwx2Time_FlatMCDMC\" or "
                             "\"MCDMCww2wxwx2Time_FlatAvgMCDMC\" or \"MCDMCww2wxwx2Time_IndvlFlatAvgMCDMC\"  type.\n"
                             "If n > 0 is specified, n samples will be consdered")

    args = parser.parse_args()
    if args.TYPE == "MCDMCTime_AvgMCDMC":
        print("Please use Code/ParameterTuning-3D/combineSamples.py in parameter-tuning-MC project.")
        print("This functionality has been deprecated")
        sys.exit(0)
        # prepareMCdata(args)
    elif args.TYPE == "MCDMCTime_FlatMCDMC":
        prepareFlatMCdata(args)
    elif args.TYPE == "MCDMCww2wxwx2Time_FlatAvgMCDMC":
        createFlatAveragedData(args)
    elif args.TYPE == "MCDMCww2wxwx2Time_IndvlFlatAvgMCDMC":
        createIndividualFlatAveragedData(args)

