import json
import numpy as np
import argparse

import apprentice

def mkCov(yerrs):
    import numpy as np
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

def run_chi2_optimization(valfile,errfile,expdatafile,wtfile,chi2resultoutfile,pstarfile,debug):
    # print("Starting chi2 optimization --")
    import sys
    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)

    res = IO.minimize(5,10)
    SCLR = IO._AS._RA[0]
    outputdata = {
        "x": res['x'].tolist(),
        "fun" : res['fun'],
        "scaler":SCLR.asDict
    }
    with open(chi2resultoutfile,'w') as f:
        json.dump(outputdata,f,indent=4)

    outds = {
        "parameters": [outputdata['x']]
    }
    if debug: print("\\SP amin \t= {}".format(["%.3f"%(c) for c in res['x']]))
    with open(pstarfile,'w') as f:
        json.dump(outds,f,indent=4)


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve TR Subproblem',
                                     formatter_class=SaneFormatter)
    parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
                        help="Value approximation file name (JSON)")
    parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
                        help="Error approximation file name (JSON)")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")
    parser.add_argument("--chi2resultfile", dest="CHI2RESULTFILE", type=str, default=None,
                        help="Result ouput file (JSON)")
    parser.add_argument("--pstarfile", dest="PSTARFILE", type=str, default=None,
                        help="p^* parameter outfile (JSON)")
    parser.add_argument("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                        help="Turn on some debug messages")

    args = parser.parse_args()
    run_chi2_optimization(
        args.VALAPPFILE,
        args.ERRAPPFILE,
        args.EXPDATA,
        args.WEIGHTS,
        args.CHI2RESULTFILE,
        args.PSTARFILE,
        args.DEBUG
    )

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass