import argparse
import json
import numpy as np

def buildInterpolationPoints(algoparams,paramfileName,iterationNo,outfile):

    ############################################################
    # Step 0: Get relevent algorithm parameters and past parameter
    # vectors
    ############################################################
    with open(algoparams,'r') as f:
        ds = json.load(f)

    tr_radius = ds['tr']['radius']
    tr_center = ds['tr']['center']
    N_p = ds['N_p']
    dim = ds['dim']
    point_min_dist = ds['point_min_dist']
    parambounds = ds['param_bounds'] if "param_bounds" in ds and ds['param_bounds'] is not None else None

    # prevparams = None
    # for iter in range(iterationNo):
    #     fname = paramfileName + "_k{}.json".format(iter)
    #     with open(fname,'r') as f:
    #         ds = json.load(f)
    #     prevparams = np.vstack([pv for pv in ds['parameters']])

    ############################################################
    # Step 1: find out which currently existing points are within
    # the radius
    ############################################################
    # if prevparams is not None:
    #     prevParamAccept = [False]*len(prevparams)
    #     for pno, p in enumerate(prevparams):
    #         distarr = [np.abs(p[vno] - tr_center[vno]) for vno in range(dim)]
    #         infn = max(distarr)
    #         prevParamAccept[pno] = infn <= tr_radius
    #     # print(prevParamAccept)
    #     # print(len(prevparams[prevParamAccept]))
    #     np_from_prev_points = len(prevparams[prevParamAccept])
    #     # print(np_from_prev_points)
    #
    #     np_remain = N_p - np_from_prev_points
    #     # np.random.seed(seed)
    # else:
    #     np_remain = N_p
    np_remain = N_p
    # print(np_remain)
    newparams = None
    if parambounds is not None:
        minarr = [max(tr_center[d] - tr_radius,parambounds[d][0]) for d in range(dim)]
        maxarr = [min(tr_center[d] + tr_radius,parambounds[d][1]) for d in range(dim)]
    else:
        minarr = [tr_center[d] - tr_radius for d in range(dim)]
        maxarr = [tr_center[d] + tr_radius for d in range(dim)]
    print("TR bounds \t= {}".format([["%.3f"%a,"%.3f"%b] for a,b in zip(minarr,maxarr)]))
    while np_remain >0:
        ############################################################
        # Step 2: get the remaining points needed (doing uniform random
        # for now)
        ############################################################

        Xperdim = ()
        for d in range(dim):
            Xperdim = Xperdim + (np.random.rand(np_remain, ) *
                                 (maxarr[d] - minarr[d]) + minarr[d],)  # Coordinates are generated in [MIN,MAX]

        Xnew = np.column_stack(Xperdim)

        ############################################################
        # Step 3: Make sure all points are at least a certain distance
        # from each other. If not, go to step 2 and repeat
        ############################################################

        for xn in Xnew:
            newparamsAccept = [True]
            newparamsAccept2 = [True]
            # if prevparams is not None:
            #     newparamsAccept = [False] * len(prevparams[prevParamAccept])
            #     for xno,xo in enumerate(prevparams[prevParamAccept]):
            #         distarr = [np.abs(xn[vno] - xo[vno]) for vno in range(dim)]
            #         infn = max(distarr)
            #         newparamsAccept[xno] = infn >= point_min_dist
            if newparams is not None:
                newparamsAccept2 = [False] * len(newparams)
                for xno,xo in enumerate(newparams):
                    distarr = [np.abs(xn[vno] - xo[vno]) for vno in range(dim)]
                    infn = max(distarr)
                    newparamsAccept2[xno] = infn >= point_min_dist

            if all(newparamsAccept) and all(newparamsAccept2):
                if newparams is not None:
                    newparams = np.concatenate((newparams, np.array([xn])))
                else:
                    newparams = np.array([xn])
                np_remain -= 1

            if np_remain == 0:
                break

    ############################################################
    # Step 4: Output all the new points to be given to the problem
    # to run the simulation on
    ############################################################
    ds = {
        "parameters":newparams.tolist()
    }
    # if prevparams is not None:
    #     ds["prevparameters"] = prevparams[prevParamAccept].tolist()
    # print(ds)
    with open(outfile,'w') as f:
        json.dump(ds, f, indent=4)


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("-p", dest="PREVPARAMSFN", type=str, default=[], nargs='+',
                        help="Previous parameters file name e.g., main_new_params_N_p")
    parser.add_argument("--iter", dest="ITER", type=int, default=0,
                        help="Current iteration number")
    parser.add_argument("-o", dest="OUTFILE", type=str, default=None,
                        help="Output file")

    args = parser.parse_args()
    buildInterpolationPoints(
        args.ALGOPARAMS,
        args.PREVPARAMSFN,
        args.ITER,
        args.OUTFILE
    )
