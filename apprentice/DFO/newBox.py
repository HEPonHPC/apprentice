import argparse
import apprentice
import h5py
import json
import numpy as np
from shutil import copyfile

def tr_update(currIterationNo,algoparams,valfile,errfile,expdatafile,wtfile,
              kpstarfile,kMCout,kp1pstarfile,kp1MCout):
    with open(algoparams, 'r') as f:
        algoparamds = json.load(f)
    gradcond = algoparamds['tr']['gradientCondition']
    tr_radius = algoparamds['tr']['radius']
    with open(kpstarfile, 'r') as f:
        ds = json.load(f)
    kpstar = ds['parameters'][0]
    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)
    if gradcond=="NO":
        kDATA = apprentice.io.readH5(kMCout)
        idx = [i for i in range(len(kDATA))]
        with h5py.File(kMCout, "r") as f:
            tmp = f.get("index")[idx]
        mcbinids = [t.decode() for t in tmp]
        kp1DATA = apprentice.io.readH5(kp1MCout)

        with open (kp1pstarfile,'r') as f:
            ds = json.load(f)
        kp1pstar = ds['parameters'][0]

        chi2_ra_k = IO.objective(kpstar)
        chi2_ra_kp1 = IO.objective(kp1pstar)

        chi2_mc_k = 0.
        chi2_mc_kp1 = 0.
        # print(mcbinids)
        # print(IO._binids)
        for mcnum, (_X, _Y, _E) in enumerate(kDATA):
            if mcbinids[mcnum] in IO._binids:
                ionum = IO._binids.index(mcbinids[mcnum])
                # print(_Y[0], IO._Y[ionum])
                chi2_mc_k += IO._W2[ionum]*((_Y[0]-IO._Y[ionum])**2/(_E[0]**2+IO._E[ionum]**2))
            else:
                continue

        for mcnum, (_X, _Y, _E) in enumerate(kp1DATA):
            if mcbinids[mcnum] in IO._binids:
                ionum = IO._binids.index(mcbinids[mcnum])
                # print(_Y[0], IO._Y[ionum])
                chi2_mc_kp1 += IO._W2[ionum]*((_Y[0]-IO._Y[ionum])**2/(_E[0]**2+IO._E[ionum]**2))
            else:
                continue
        # print("chi2_ra_k=\t{}\nchi2_ra_kp1=\t{}\nchi2_mc_k=\t{}\nchi2_mc_kp1=\t{}\n".format(chi2_ra_k,chi2_ra_kp1,chi2_mc_k,chi2_mc_kp1))

        print("chi2/ra k\t= %.3f" % (chi2_ra_k))
        print("chi2/ra k+1\t= %.3f" % (chi2_ra_kp1))
        print("chi2/mc k\t= %.3f" % (chi2_mc_k))
        print("chi2/mc k+1\t= %.3f" % (chi2_mc_kp1))

        rho = (chi2_mc_k - chi2_mc_kp1) / (chi2_ra_k - chi2_ra_kp1)
        # print("rho={}".format(rho))

        eta = algoparamds['tr']['eta']
        sigma = algoparamds['tr']['sigma']
        tr_maxradius = algoparamds['tr']['maxradius']

        # grad = IO.gradient(kpstar)
        print("rho k\t\t= %.3f" % (rho))
        if rho < eta :
                # or np.linalg.norm(grad) <= sigma * tr_radius:
            # print("rho < eta New point rejected")
            tr_radius /=2
            curr_p = kpstar
            trradmsg = "TR radius halved"
            trcentermsg = "TR center remains the same"
            copyfile(kpstarfile,kp1pstarfile)
            copyfile(kMCout,kp1MCout)
        else:
            # print("rho >= eta. New point accepted")
            tr_radius = min(tr_radius*2,tr_maxradius)
            curr_p = kp1pstar
            # copyfile(src, dst)
            # copyfile(kp1pstarfile,kpstarfile)
            # copyfile(kp1MCout,kMCout)
            trradmsg = "TR radius doubled"
            trcentermsg = "TR center moved to the SP amin"
    else:
        # print("gradient condition failed")
        tr_radius /= 2
        curr_p = kpstar
        trradmsg = "TR radius halved"
        trcentermsg = "TR center remains the same"
        copyfile(kpstarfile,kp1pstarfile)
        copyfile(kMCout,kp1MCout)
    # put  tr_radius and curr_p in radius and center and write to algoparams
    algoparamds['tr']['radius'] = tr_radius
    algoparamds['tr']['center'] = curr_p
    print("\Delta k+1 \t= %.2f (%s)"%(tr_radius,trradmsg))

    print("P k+1 \t\t= {} ({})".format(["%.3f"%(c) for c in curr_p],trcentermsg))

    # Stopping condition
    # get parameters
    max_iteration = algoparamds['max_iteration']
    min_gradientNorm = algoparamds['min_gradientNorm']
    max_simulationBudget = algoparamds['max_simulationBudget']

    # get budget
    simulationbudgetused = algoparamds['simulationbudgetused']

    # get gradient of model at current point
    IO._AS.setRecurrence(curr_p)
    IO._EAS.setRecurrence(curr_p)
    grad = IO.gradient(curr_p)

    status = "CONTINUE"
    if np.linalg.norm(grad) <= min_gradientNorm:
        status = "STOP"
        print("STOP\t\t= Norm of the gradient too small {}".format(np.linalg.norm(grad)))
    if currIterationNo >= max_iteration-1:
        status = "STOP"
        print("STOP\t\t= Max iterations reached")
    if simulationbudgetused >= max_simulationBudget:
        status = "STOP"
        print("STOP\t\t= Simulation budget depleted")
    print(status)
    algoparamds['status'] = status
    with open(algoparams,'w') as f:
        json.dump(algoparamds,f,indent=4)



    with open(algoparams,'w') as f:
        json.dump(algoparamds,f,indent=4)


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TR Update',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
                        help="Value approximation file name")
    parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
                        help="Error approximation file name")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file")
    parser.add_argument("--km1pstarfile", dest="KM1PSTARFILE", type=str, default=None,
                        help="p^* parameter file from iteration k-1")
    parser.add_argument("--kpstarfile", dest="KPSTARFILE", type=str, default=None,
                        help="p^* parameter outfile from iteration k")
    parser.add_argument("--km1MCout", dest="KM1MCOUT", type=str, default=None,
                        help="MC OUT H5 from iteration k-1")
    parser.add_argument("--kMCout", dest="KMCOUT", type=str, default=None,
                        help="MC OUT H5 from iteration k")

    args = parser.parse_args()