import argparse
import apprentice
from sampleSet import buildInterpolationPoints
from problem import problem_main_program
from approx import run_approx
from chi2 import run_chi2_optimization
from newBox import tr_update
import numpy as np
import json
import sys,os
from shutil import copyfile
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
"""
ddd=X2_2D_1bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights 

ddd=X2_2D_3bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights 

ddd=X2_2D_3bin_notglobal; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights 

TEST FOR ORCUN
ddd=X2_2D_3bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights -c ../../../log/DFO/P/process.dat -v 
ddd=SumOfDiffPowers_2D_3bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights -c ../../../log/DFO/P/process.dat -v


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON)")
    parser.add_argument("-s", dest="SEED", type=int, default=2376762,
                        help="Random seed")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")
    parser.add_argument("-d", dest="WD", type=str, default=None,
                        help="Create and use path (STR) as working Directory")
    parser.add_argument("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                        help="Turn on some debug messages")

    args = parser.parse_args()
    np.random.seed(args.SEED)

    os.makedirs(args.WD,exist_ok=True)
    newparams_Np = args.WD + "/new_params_Np"
    prevparams_Np = args.WD + "/prev_params_Np"
    newparams_1 = args.WD + "/new_params_1"
    MCout_Np = args.WD + "/MCout_Np"
    MCout_1 = args.WD + "/MCout_1"
    valapproxfile = args.WD + "/valapprox"
    errapproxfile = args.WD + "/errapprox"
    resultoutfile = args.WD + "/chi2result"
    pythiadir_Np = args.WD + "/pythia_Np"
    pythiadir_1 = args.WD + "/pythia_1"

    algoparamsfile = os.path.join(args.WD, "algoparams.json")
    assert(algoparamsfile!=args.ALGOPARAMS)
    copyfile(args.ALGOPARAMS,algoparamsfile)

    k=0
    with open(args.EXPDATA, 'r') as f:
        expdata = json.load(f)
    binids = [b for b in expdata]
    while True:
        newparams_Np_k = newparams_Np + "_k{}.json".format(k)
        prevparams_Np_k = prevparams_Np + "_k{}.json".format(k)
        newparams_1_kp1 = newparams_1 + "_k{}.json".format(k + 1)
        newparams_1_k = newparams_1 + "_k{}.json".format(k)
        MCout_Np_k = MCout_Np + "_k{}.h5".format(k)
        MCout_1_k = MCout_1 + "_k{}.h5".format(k)
        MCout_1_kp1 = MCout_1 + "_k{}.h5".format(k + 1)
        valapproxfile_k = valapproxfile + "_k{}.json".format(k)
        errapproxfile_k = errapproxfile + "_k{}.json".format(k)
        resultoutfile_k = resultoutfile + "_k{}.json".format(k)
        pythiadir_Np_k = pythiadir_Np + "_k{}".format(k)
        pythiadir_1_k = pythiadir_1 + "_k{}".format(k)
        pythiadir_1_kp1 = pythiadir_1 + "_k{}".format(k+1)

        if k==0:
            with open(algoparamsfile, 'r') as f:
                algoparamds = json.load(f)
            tr_center = algoparamds['tr']['center']
            parambounds = algoparamds['param_bounds'] if "param_bounds" in algoparamds and \
                                                         algoparamds['param_bounds'] is not None \
                                                        else None
            paramnames = algoparamds["param_names"]
            dim = algoparamds['dim']
            if args.DEBUG:
                print("\n#####################################")
                print("Initially")
                print("#####################################")
                print("\Delta_1 \t= {}".format(algoparamds['tr']['radius']))
                print("N_p \t\t= {}".format(algoparamds['N_p']))
                print("dim \t\t= {}".format(dim))
                print("|B| \t\t= {}".format(len(binids)))
                print("P_1 \t\t= {}".format(algoparamds['tr']['center']))
            if parambounds is not None:
                for d in range(dim):
                    if parambounds[d][0] > tr_center[d] or tr_center[d] > parambounds[d][1]:
                        raise Exception("Starting TR center along dimension {} is not within parameter bound "
                                        "[{}, {}]".format(d+1,parambounds[d][0],parambounds[d][1]))
                if args.DEBUG: print("Phy bounds \t= {}".format(parambounds))
            else:
                if args.DEBUG: print("Phy bounds \t= {}".format(None))
            outds = {
                "parameters": [tr_center]
            }
            with open(newparams_1_k, 'w') as f:
                json.dump(outds, f, indent=4)
            apprentice.tools.writePythiaFiles(args.PROCESSCARD,paramnames,[tr_center],pythiadir_1_k)
            problem_main_program(algoparams=algoparamsfile,
                                 paramfile=newparams_1_k,
                                 pythiadir=pythiadir_1_k,
                                 binids=binids,
                                 outfile=MCout_1_k,
                                 debug=args.DEBUG)

        if args.DEBUG:
            print("\n#####################################")
            print("Starting iteration {}".format(k + 1))
            print("#####################################")

        buildInterpolationPoints(algoparams=algoparamsfile,iterationNo=k,
                                 processcard=args.PROCESSCARD,outdir=pythiadir_Np_k,
                                 newparamoutfile=newparams_Np_k, debug=args.DEBUG)

        problem_main_program(algoparams=algoparamsfile,
                             paramfile=newparams_Np_k,
                             pythiadir=pythiadir_Np_k,
                             binids=binids,
                             outfile=MCout_Np_k,
                             debug=args.DEBUG)

        run_approx(algoparams=algoparamsfile,
                   interpolationdatafile=MCout_Np_k,
                   valoutfile=valapproxfile_k,
                   erroutfile=errapproxfile_k,
                   expdatafile=args.EXPDATA,
                   wtfile=args.WEIGHTS,
                   debug=args.DEBUG)

        with open(algoparamsfile, 'r') as f:
            algoparamds = json.load(f)
        gradcond = algoparamds['tr']['gradientCondition']

        if gradcond == "NO":
            with open(algoparamsfile, 'r') as f:
                algoparamds = json.load(f)
            paramnames = algoparamds["param_names"]
            run_chi2_optimization(algoparams=args.ALGOPARAMS,
                                  proccardfile=args.PROCESSCARD,
                                  valfile=valapproxfile_k,
                                  errfile=errapproxfile_k,
                                  expdatafile=args.EXPDATA,
                                  wtfile=args.WEIGHTS,
                                  chi2resultoutfile=resultoutfile_k,
                                  pstarfile=newparams_1_kp1,
                                  pythiadir=pythiadir_1_kp1,
                                  debug=args.DEBUG)

            problem_main_program(algoparams=algoparamsfile,
                                 paramfile=newparams_1_kp1,
                                 pythiadir=pythiadir_1_kp1,
                                 binids=binids,
                                 outfile=MCout_1_kp1,
                                 debug=args.DEBUG)

        tr_update(currIterationNo=k,
                  algoparams=algoparamsfile,
                  valfile=valapproxfile_k,
                  errfile=errapproxfile_k,
                  expdatafile=args.EXPDATA,
                  wtfile=args.WEIGHTS,
                  kpstarfile=newparams_1_k,
                  kMCout=MCout_1_k,
                  kp1pstarfile=newparams_1_kp1,
                  kp1MCout=MCout_1_kp1,
                  debug=args.DEBUG)
        k += 1
        with open(algoparamsfile, 'r') as f:
            algoparamds = json.load(f)
        if algoparamds['status'] == "STOP":
            break
        # exit(1)




