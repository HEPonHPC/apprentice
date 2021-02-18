import h5py
import apprentice
import json
import sys
import argparse
import numpy as np

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
def run_approx(algoparams,interpolationdatafile,valoutfile, erroutfile,expdatafile,wtfile,debug):
    # print("Starting approximation --")
    # print("CHANGE ME TO THE PARALLEL VERSION")
    assert (erroutfile != interpolationdatafile)
    assert (valoutfile != interpolationdatafile)
    DATA = apprentice.io.readH5(interpolationdatafile)
    # print(DATA)
    pnames = apprentice.io.readPnamesH5(interpolationdatafile, xfield="params")
    idx = [i for i in range(len(DATA))]
    valapp = []
    errapp = []

    # S = apprentice.Scaler(DATA[0][0])  # Let's assume that all X are the same for simplicity
    # print("Halfway reporting: before generating the output file --")
    for num, (_X, _Y, _E) in enumerate(DATA):
        # print(_X)
        # print(_Y)
        try:
            # print("\n\n\n\n")
            # print(_X,_Y,_E)
            valapp.append(apprentice.RationalApproximation(_X, _Y, order=(2, 0), pnames=pnames))
            errapp.append(apprentice.RationalApproximation(_X, _E, order=(1, 0), pnames=pnames))
        except AssertionError as error:
            print(error)
    # What do these do? Are the next 4 lines required
    # S.save("{}.scaler".format(valoutfile))
    # S.save("{}.scaler".format(erroutfile))
    # S.save(valoutfile)
    # S.save(erroutfile)

    # This reads the unique identifiers of the bins
    with h5py.File(interpolationdatafile, "r") as f:
        binids = f.get("index")[idx]

    JD = {x.decode(): y.asDict for x, y in zip(binids, valapp)}
    with open(valoutfile, "w") as f:
        json.dump(JD, f)

    JD = {x.decode(): y.asDict for x, y in zip(binids, errapp)}
    with open(erroutfile, "w") as f:
        json.dump(JD, f)

    # print("Done --- approximation of {} objects written to {} and {}".format(
    #         len(idx), valoutfile, erroutfile))

    with open(algoparams,'r') as f:
        algoparamds = json.load(f)

    tr_center = algoparamds['tr']['center']
    tr_radius = algoparamds['tr']['radius']
    sigma = algoparamds['tr']['sigma']

    # print("BYE from approx")
    sys.stdout.flush()

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valoutfile,
                                            erroutfile)
    IO._AS.setRecurrence(tr_radius)
    IO._EAS.setRecurrence(tr_radius)
    grad = IO.gradient(tr_center)
    # print(np.linalg.norm(grad),sigma, tr_radius)
    if np.linalg.norm(grad) <= sigma * tr_radius:
        algoparamds['tr']['gradientCondition'] = "YES"
    else: algoparamds['tr']['gradientCondition'] = "NO"
    if debug:print("||grad|| \t= %.3f"%(np.linalg.norm(grad)))
    with open(algoparams,'w') as f:
        json.dump(algoparamds,f,indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct Model',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON)")
    parser.add_argument("-i", dest="INTERPOLATIONDATAFILE", type=str, default=None,
                        help="Interpolation data (MC HDF5) file")
    parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
                        help="Value approximation output file name (JSON)")
    parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
                        help="Error approximation output file name (JSON)")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")
    parser.add_argument("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                        help="Turn on some debug messages")

    args = parser.parse_args()

    run_approx(
        args.ALGOPARAMS,
        args.INTERPOLATIONDATAFILE,
        args.VALAPPFILE,
        args.ERRAPPFILE,
        args.EXPDATA,
        args.WEIGHTS,
        args.DEBUG
    )