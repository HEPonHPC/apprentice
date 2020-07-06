import sys,os
import argparse
import numpy as np
import csv

def prepareMCdata(args):
    import pandas as pd
    avgMC = np.array([])
    avgDMC = np.array([])
    X = np.array([])
    for fno,file in enumerate(args.DATAFILES):
        data = pd.read_csv(file, header=None)
        D = data.values
        MC = D[:, -3]
        DeltaMC = D[:, -2]
        if fno == 0:
            X = np.array(D[:, :-3])
            avgMC = np.zeros(len(MC))
            avgDMC = np.zeros(len(MC))
        avgMC+=MC
        avgDMC+=DeltaMC
    avgMC = np.atleast_2d(avgMC/len(args.DATAFILES))
    avgDMC = np.atleast_2d(avgDMC / len(args.DATAFILES))
    # print(X)
    # print(avgMC)
    # print(avgDMC)
    dir = os.path.dirname(args.OUTFILE)
    os.makedirs(dir,exist_ok=True)
    np.savetxt(args.OUTFILE, np.hstack((X, avgMC.T,avgDMC.T)), delimiter=",")

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
    parser.add_argument("-t", "--type", dest="TYPE", type=str, default="MCDMCTime_AvgMCDMC",
                               choices=["MCDMCTime_AvgMCDMC"], help="Type")

    args = parser.parse_args()
    if args.TYPE == "MCDMCTime_AvgMCDMC":
        prepareMCdata(args)

