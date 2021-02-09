import argparse
from sampleSet import buildInterpolationPoints
from problem import problem_main_program
from approx import run_approx
from chi2 import run_chi2_optimization
from newBox import tr_update
import numpy as np
import sys,os
from shutil import copyfile
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
"""
ddd=X2_2D_1bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights -b X2#1

ddd=X2_2D_3bin; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights -b X2#1 X2#2 X2#3

ddd=X2_2D_3bin_notglobal; python main.py -a ../../../log/DFO/P/$ddd/algoparams_bk.json -s 876 -d /tmp/DFO/$ddd -e ../../../log/DFO/P/$ddd/data.json -w ../../../log/DFO/P/$ddd/weights -b X2#1 X2#2 X2#3
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("-s", dest="SEED", type=int, default=2376762,
                        help="Random seed")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file")
    parser.add_argument("-b", dest="BINIDS", type=str, default=[], nargs='+',
                        help="Bin ids Shekel#1 or X2#1 and so on")
    parser.add_argument("-d", dest="WD", type=str, default=None,
                        help="Working Directory")

    args = parser.parse_args()
    np.random.seed(args.SEED)

    os.makedirs(args.WD,exist_ok=True)
    newparams_Np = args.WD + "/new_params_Np"
    newparams_1 = args.WD + "/new_params_1"
    MCout_Np = args.WD + "/MCout_Np"
    MCout_1 = args.WD + "/MCout_1"
    valapproxfile = args.WD + "/valapprox"
    errapproxfile = args.WD + "/errapprox"
    resultoutfile = args.WD + "/chi2result"
    k=0
    newparams_Np_k = newparams_Np+"_k{}.json".format(k)
    newparams_1_kp1 = newparams_1+"_k{}.json".format(k+1)
    newparams_1_k = newparams_1+"_k{}.json".format(k)
    MCout_Np_k = MCout_Np+"_k{}.h5".format(k)
    MCout_1_k = MCout_1 + "_k{}.h5".format(k)
    MCout_1_kp1 = MCout_1+"_k{}.h5".format(k+1)
    valapproxfile_k = valapproxfile+"_k{}.json".format(k)
    errapproxfile_k = errapproxfile+"_k{}.json".format(k)
    resultoutfile_k = resultoutfile+"_k{}.json".format(k)

    algoparamsfile = os.path.join(args.WD, "algoparams.json")
    assert(algoparamsfile!=args.ALGOPARAMS)
    copyfile(args.ALGOPARAMS,algoparamsfile)

    for k in range(5):
        if k==0:
            import json
            with open(algoparamsfile, 'r') as f:
                algoparamds = json.load(f)
            tr_center = algoparamds['tr']['center']
            parambounds = algoparamds['param_bounds'] if "param_bounds" in algoparamds and \
                                                         algoparamds['param_bounds'] is not None \
                                                        else None
            dim = algoparamds['dim']
            print("\n#####################################")
            print("Initially")
            print("#####################################")
            print("\Delta_1 \t= {}".format(algoparamds['tr']['radius']))
            print("N_p \t\t= {}".format(algoparamds['N_p']))
            print("dim \t\t= {}".format(dim))
            print("|B| \t\t= {}".format(len(args.BINIDS)))
            print("P_1 \t\t= {}".format(algoparamds['tr']['center']))
            if parambounds is not None:
                for d in range(dim):
                    if parambounds[d][0] > tr_center[d] or tr_center[d] > parambounds[d][1]:
                        raise Exception("Starting TR center along dimension {} is not within parameter bound "
                                        "[{}, {}]".format(d+1,parambounds[d][0],parambounds[d][1]))
                print("Phy bounds \t= {}".format(parambounds))
            else:
                print("Phy bounds \t= {}".format(None))
            outds = {
                "parameters": [tr_center]
            }
            with open(newparams_1_k, 'w') as f:
                json.dump(outds, f, indent=4)
            problem_main_program(algoparamsfile, newparams_1_k, args.BINIDS, MCout_1_k)

        print("\n#####################################")
        print("Starting iteration {}".format(k + 1))
        print("#####################################")

        buildInterpolationPoints(algoparamsfile,newparams_Np,k,newparams_Np_k)
        problem_main_program(algoparamsfile,newparams_Np_k,args.BINIDS,MCout_Np_k)
        run_approx(algoparamsfile,MCout_Np_k,valapproxfile_k,errapproxfile_k,
                   args.EXPDATA,args.WEIGHTS)

        with open(algoparamsfile, 'r') as f:
            algoparamds = json.load(f)
        gradcond = algoparamds['tr']['gradientCondition']

        if gradcond == "NO":
            run_chi2_optimization(algoparamsfile, valapproxfile_k,errapproxfile_k, args.EXPDATA,args.WEIGHTS,
                              resultoutfile_k,newparams_1_kp1)
            problem_main_program(algoparamsfile, newparams_1_kp1, args.BINIDS, MCout_1_kp1)

        tr_update(algoparamsfile, valapproxfile_k,errapproxfile_k, args.EXPDATA,args.WEIGHTS,
                  newparams_1_k, MCout_1_k, newparams_1_kp1, MCout_1_kp1)

        # exit(1)




