import numpy as np
import json
import argparse
import sys

# Number of local minima
m = 10
# Number of parameter dimensions
d = 4
# Order of the local minima
beta = (1/m) * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
# All the local minima: d x m dimensional matrix
C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])

# https://www.sfu.ca/~ssurjano/shekel.html
def shekelObjective(x):
    outer = 0
    for i in range(m):
        bi = beta[i]
        inner = 0
        for j in range(d):
            inner += (x[j] - C[j, i]) ** 2
        outer = outer + 1 / (inner + bi)
    return -1 * outer

# https://www.sfu.ca/~ssurjano/sumpow.html
def x2Objective(x):
    sum = 0
    for ii in range(len(x)):
        xi = x[ii]
        new = (abs(xi)) ** (ii + 1);
        sum = sum + new
    return sum

def runSimulation(p,fidelity,problemname,factor=1):
    """
    Run simulation
    :param x: parameter
    :param n: Fidelity
    :return:
    """
    if problemname == "Shekel":
        Y = np.random.normal(factor*(shekelObjective(p)),1/np.sqrt(fidelity),1)
    elif problemname == "X2":
        Y = np.random.normal(factor*(x2Objective(p)), 1 / np.sqrt(fidelity), 1)
    elif problemname == "Hybrid":
        Y = np.random.normal(factor*(x2Objective(p)+shekelObjective(p)),
                              1 / np.sqrt(fidelity), 1)
    else: raise Exception("Problem name {} unknown".format(problemname))
    E = [1.]
    return Y,E

def problem_main_program(algoparams,paramfile,binids,outfile):
    with open(algoparams,'r') as f:
        ds = json.load(f)
    param_names = ds["param_names"]
    fidelity = ds["fidelity"]
    dim = ds['dim']

    with open(paramfile,'r') as f:
        ds = json.load(f)
    P = ds['parameters']

    HNAMES = np.array([b.split("#")[0]  for b in binids])
    FACTOR = np.array([int(b.split("#")[1])  for b in binids])
    BNAMES = binids
    PARAMS = {}
    for pno,p in enumerate(P):
        ppp = {}
        for d in range(dim):
            ppp[param_names[d]] = p[d]
        PARAMS[str(pno)] = ppp
    pnames = PARAMS[list(PARAMS.keys())[0]].keys()
    runs = sorted(list(PARAMS.keys()))
    vals = []
    errs = []
    for bno, b in enumerate(binids):
        vals.append([
            runSimulation(p,fidelity=fidelity,problemname=HNAMES[bno],factor=FACTOR[bno])[0][0]
            for p in P
        ])
        errs.append([
            runSimulation(p, fidelity=fidelity, problemname=HNAMES[bno], factor=FACTOR[bno])[1][0]
            for p in P
        ])


    # print("##########")
    # print(vals)
    # print(errs)
    # print("##########")
    import h5py
    f = h5py.File(outfile, "w")

    # print(pnames)
    # print(runs)

    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'), compression=4)
    f.create_dataset("runs", data=np.char.encode(runs, encoding='utf8'), compression=4)
    pset = f.create_dataset("params", data=np.array([list(PARAMS[r].values()) for r in runs]),
                            compression=9)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]
    f.create_dataset("values", data=vals, compression=4)
    f.create_dataset("errors", data=errs, compression=4)
    f.close()

    # print("Done. Output written to %s" % outfile)
    sys.stdout.flush()


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Simulation (with noise added)',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("-p", dest="CURRENTPARAMS", type=str, default=None,
                        help="Current parameters in JSON")
    parser.add_argument("-b", dest="BINIDS", type=str, default=[], nargs='+',
                        help="Bin ids Shekel#1 or X2#1 and so on")
    parser.add_argument("-o", dest="OUTFILE", type=str, default=None,
                        help="Output file")

    args = parser.parse_args()

    print(runSimulation(P=[[4, 4, 4, 4]], fidelity=1000, problemname="Shekel"))
