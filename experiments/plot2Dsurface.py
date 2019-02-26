import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os
from mpl_toolkits.mplot3d import Axes3D

def plot2Dsurface(infile,testfile,folder, desc,bottom_or_all):

    import json
    if infile:
        with open(infile, 'r') as fn:
            datastore = json.load(fn)
    dim = datastore['dim']
    if(dim != 2):
        raise Exception("plot2Dsurface can only handle dim = 2")
    m = datastore['m']
    n = datastore['n']
    ts = datastore['trainingscale']
    trainingsize = datastore['trainingsize']


    X, Y = readData(testfile)
    if(bottom_or_all == "bottom"):
        testset = [i for i in range(trainingsize,len(X_test))]
        X_test = X[testset]
        Y_test = Y[testset]
    else:
        X_test = X
        Y_test = Y

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')

    outfilepng = "%s/plots/P2d_%s_p%d_q%d_ts%s_2Dsurface.png"%(folder,desc,m,n,ts)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    ax.plot3D(X_test[:,0],X_test[:,1], Y[:],"r.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('Original data', fontsize = 13)

    rappsip = RationalApproximationSIP(datastore)
    Y_pred = rappsip.predictOverArray(X_test)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot3D(X_test[:,0],X_test[:,1], Y_pred[:],"b.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('Predicted data', fontsize = 13)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot3D(X_test[:,0],X_test[:,1], np.absolute(Y_pred-Y_test),"g.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('|Predicted - Original|', fontsize = 13)

    l1 = np.sum(np.absolute(Y_pred-Y_test))
    l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
    linf = np.max(np.absolute(Y_pred-Y_test))
    nnz = tools.numNonZeroCoeff(rappsip,1e-6)
    fig.suptitle("%s. m = %d, n = %d, ts = %d (%s). l1 = %.4f, l2 = %.4f, linf = %.4f, nnz = %d, l2/nnz = %f"%(desc,m,n,trainingsize,ts,l1,l2,linf,nnz,l2/nnz))

    plt.savefig(outfilepng)

# python plot2Dsurface.py f21_2x/out/f21_2x_p12_q12_ts2x.json ../benchmarkdata/f21_test.txt f21_2x f21_2x all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=6:
        print("Usage: {} infile testfile folder fndesc bottom_or_all".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))

    plot2Dsurface(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5])
###########
