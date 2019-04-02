import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os
from mpl_toolkits.mplot3d import Axes3D

def getbox(f):
    box = []
    if(f=="f7"):
        box = [[0,1],[0,1]]
    elif(f=="f10" or f=="f19"):
        box = [[-1,1],[-1,1],[-1,1],[-1,1]]
    elif(f=="f17"):
        box = [[80,100],[5,10],[90,93]]
    elif(f=="f18"):
        box = [[-0.95,0.95],[-0.95,0.95],[-0.95,0.95],[-0.95,0.95]]
    elif(f=="f20"):
        box = [[10**-6,4*np.pi],[10**-6,4*np.pi],[10**-6,4*np.pi],[10**-6,4*np.pi],[10**-6,4*np.pi],[10**-6,4*np.pi],[10**-6,4*np.pi]]
    elif(f=="f21"):
        box = [[10**-6,4*np.pi],[10**-6,4*np.pi]]
    else:
        box = [[-1,1],[-1,1]]
    return box


def suppresscoeffs(datastore, nnzthreshold):
    nnzthreshold = 1e-6
    for i, p in enumerate(datastore['pcoeff']):
        if(abs(p)<nnzthreshold):
            datastore['pcoeff'][i] = 0.
    if('qcoeff' in datastore):
        for i, q in enumerate(datastore['qcoeff']):
            if(abs(q)<nnzthreshold):
                datastore['qcoeff'][i] = 0.

# def calculatetesterror(Y_test,Y_pred,app,nnzthreshold):
#     nnz = float(tools.numNonZeroCoeff(app,nnzthreshold))
#     l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
#     ret =  l2 / nnz
#     return ret

def plot2Dsurface(fname, noise, m, n, ts, fndesc, papp_or_no_papp):
    noisestr = ""
    if(noise!="0"):
        noisestr = "_noisepct"+noise
    folder = "%s%s_%s"%(fname,noisestr,ts)

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')

    if(papp_or_no_papp == "papp"):
        cols = 4
    elif(papp_or_no_papp == "no_papp"): cols = 3
    else:raise Exception("papp_or_no_papp ambiguous")

    box = getbox(fname)
    if(len(box) != 2):
        print("{} cannot handle dim != 2. Box len was {}".format(sys.argv[0],len(box)))
        sys.exit(1)
    npoints = 250
    X_test1 = np.linspace(box[0][0], box[0][1], num=npoints)
    X_test2 = np.linspace(box[1][0], box[1][1], num=npoints)

    outx1 = "%s/plots/Cfnsurf_X_%s%s_p%s_q%s_ts%s.csv"%(folder, fname, noisestr,m,n,ts)
    outy = "%s/plots/Cfnsurf_Y_%s%s_p%s_q%s_ts%s.csv"%(folder, fname, noisestr,m,n,ts)


    rappsipfile = "%s/out/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,m,n,ts)
    rappfile = "%s/outra/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,m,n,ts)
    pappfile = "%s/outpa/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,m,n,ts)

    if not os.path.exists(rappsipfile):
        print("rappsipfile %s not found"%(rappsipfile))
        exit(1)

    if not os.path.exists(rappfile):
        print("rappfile %s not found"%(rappfile))
        exit(1)

    if not os.path.exists(pappfile):
        print("pappfile %s not found"%(pappfile))
        exit(1)

    rappsip = RationalApproximationSIP(rappsipfile)
    rapp = RationalApproximationONB(fname=rappfile)
    papp = PolynomialApproximation(fname=pappfile)

    if(rappsip.dim != 2):
        print("{} cannot handle dim != 2 Dim found in datastore was {}".format(sys.argv[0],rappsip.dim))
        sys.exit(1)
    from apprentice import testData
    Y_test = []
    for x in X_test1:
        for y in X_test2:
            Y_test.append(eval('testData.'+fname+'(['+str(x)+','+str(y)+'])'))

    Y_pred_rappsip = []
    for x in X_test1:
        for y in X_test2:
            Y_pred_rappsip.append(rappsip.predict([x,y]))

    Y_pred_rapp = []
    for x in X_test1:
        for y in X_test2:
            Y_pred_rapp.append(rapp([x,y]))

    Y_pred_papp = []
    if(papp_or_no_papp == "papp"):
        for x in X_test1:
            for y in X_test2:
                Y_pred_papp.append(papp([x,y]))

    np.savetxt(outx1,np.stack((X_test1,X_test2),axis=1), delimiter=",")

    def removezeros(YYY):
        for num,y in enumerate(YYY):
            if(y==0):
                # print(y, num)
                lower = 0
                upper = len(YYY)-1

                index = num
                index -=1
                while(index >= 0):
                    if(YYY[index]!=0):
                        lower = index
                        break
                    index -=1

                index = num
                index +=1
                while(index < len(YYY)):
                    if(YYY[index]!=0):
                        upper = index
                        break
                    index +=1
                YYY[num] = (YYY[lower] + YYY[upper])/2
                # print("became")
                # print(YYY[num], num)
        return YYY
    Y_pred_papp_diff = np.absolute(np.array(Y_pred_papp)-np.array(Y_test))
    Y_pred_papp_diff = removezeros(Y_pred_papp_diff)

    Y_pred_rapp_diff = np.absolute(np.array(Y_pred_rapp)-np.array(Y_test))
    Y_pred_rapp_diff = removezeros(Y_pred_rapp_diff)

    Y_pred_rappsip_diff = np.absolute(np.array(Y_pred_rappsip)-np.array(Y_test))
    Y_pred_rappsip_diff = removezeros(Y_pred_rappsip_diff)

    np.savetxt(outy,np.stack((np.ma.log10(Y_pred_papp_diff),np.ma.log10(Y_pred_rapp_diff),np.ma.log10(Y_pred_rappsip_diff)),axis=1), delimiter=",")

    minyval = np.inf
    maxyval = 0
    arr1 = np.array([min(Y_pred_rappsip),min(Y_test),min(Y_pred_rapp)])
    if(papp_or_no_papp == "papp"):
        np.append(arr1,min(Y_pred_papp))
    minyval = np.min(arr1)

    arr1 = np.array([max(Y_pred_rappsip),max(Y_test),max(Y_pred_rapp)])
    if(papp_or_no_papp == "papp"):
        np.append(arr1,max(Y_pred_papp))
    maxyval = np.max(arr1)

    import matplotlib
    import matplotlib.colors as colors
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.text as text
    cmap1 = matplotlib.cm.RdYlBu
    cmap3 = matplotlib.cm.seismic
    cmap4 = matplotlib.cm.viridis
    cmap5 = matplotlib.cm.coolwarm
    cmap6 = matplotlib.cm.magma
    cmap2='hot'
    # https://stackoverflow.com/questions/16267143/matplotlib-single-colored-colormap-with-saturation
    from matplotlib.colors import LinearSegmentedColormap
    def CustomCmap(from_rgb,to_rgb):

        # from color r,g,b
        r1,g1,b1 = from_rgb

        # to color r,g,b
        r2,g2,b2 = to_rgb

        cdict = {'red': ((0, r1, r1),
                       (1, r2, r2)),
               'green': ((0, g1, g1),
                        (1, g2, g2)),
               'blue': ((0, b1, b1),
                       (1, b2, b2))}

        ccc = LinearSegmentedColormap('custom_cmap', cdict)
        return ccc
    cmap7 = CustomCmap([0.00, 0.00, 0.00], [0.02, 0.75, 1.00])
    cmap8 = CustomCmap([1.00, 0.42, 0.04], [0.02, 0.75, 1.00])
    cmap = cmap8


    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1, cols, 1, projection='3d')
    Y_test = np.reshape(Y_test, [len(X_test1), len(X_test2)])
    im1 = ax.contour3D(X_test1, X_test2, Y_test, 100, cmap=cmap, norm = colors.SymLogNorm(linthresh=0.2, linscale=0.5,vmin=minyval, vmax=maxyval),alpha=0.8)
    ax.set_title('(a)', fontsize = 12)
    ax.set_xlabel('$x_1$', fontsize = 12)
    ax.set_ylabel('$x_2$', fontsize = 12)

    ax = fig.add_subplot(1, cols, 2, projection='3d')
    Y_pred_rappsip = np.reshape(Y_pred_rappsip, [len(X_test1), len(X_test2)])
    im1 = ax.contour3D(X_test1, X_test2, Y_pred_rappsip, 100, cmap=cmap, norm = colors.SymLogNorm(linthresh=0.2, linscale=0.5,vmin=minyval, vmax=maxyval),alpha=0.8)
    ax.set_title('(b)', fontsize = 12)
    ax.set_xlabel('$x_1$', fontsize = 12)
    ax.set_ylabel('$x_2$', fontsize = 12)

    ax = fig.add_subplot(1, cols, 3, projection='3d')
    Y_pred_rapp = np.reshape(Y_pred_rapp, [len(X_test1), len(X_test2)])
    im1 = ax.contour3D(X_test1, X_test2, Y_pred_rapp, 100, cmap=cmap, norm = colors.SymLogNorm(linthresh=0.2, linscale=0.5,vmin=minyval, vmax=maxyval),alpha=0.8)
    ax.set_title('(c)', fontsize = 12)
    ax.set_xlabel('$x_1$', fontsize = 12)
    ax.set_ylabel('$x_2$', fontsize = 12)

    if(papp_or_no_papp == "papp"):
        ax = fig.add_subplot(1, cols, 4, projection='3d')
        Y_pred_papp = np.reshape(Y_pred_papp, [len(X_test1), len(X_test2)])
        im1 = ax.contour3D(X_test1, X_test2, Y_pred_papp, 100, cmap=cmap, norm = colors.SymLogNorm(linthresh=0.2, linscale=0.5,vmin=minyval, vmax=maxyval),alpha=0.8)
        ax.set_title('(d)', fontsize = 12)
        ax.set_xlabel('$x_1$', fontsize = 12)
        ax.set_ylabel('$x_2$', fontsize = 12)

    mmm = plt.cm.ScalarMappable(cmap=cmap)
    mmm.set_array(Y_pred_rapp)
    mmm.set_clim(minyval, maxyval)
    b2=fig.colorbar(mmm)
    b2.set_label("$\\frac{p(x_1,x_2)}{q(x_1,x_2)}$", fontsize = 16)

    outfile = "%s/plots/P2d1f_%s%s_p%s_q%s_ts%s_%s.pdf"%(folder, fname, noisestr,m,n,ts,papp_or_no_papp)
    # plt.show()
    plt.savefig(outfile)
    print("open %s;"%(outfile))
    plt.close('all')

# python plot2Dsurface.py f21_2x/out/f21_2x_p12_q12_ts2x.json ../benchmarkdata/f21_test.txt f21_2x f21_2x all

if __name__ == "__main__":

    # import apprentice
    # name = "f14"
    # noisestr = "_noisepct10-3"
    # # noisestr = ""
    # trainfile = "../benchmarkdata/"+name+noisestr+".txt"
    # X, Y = readData(trainfile)
    # folder = "poletest"
    # if not os.path.exists(folder):
    #     os.mkdir(folder)
    # for m in range(1,6):
    #     for n in range(1,6):
    #         trainingsize = 2 * tools.numCoeffsRapp(2,(m,n))
    #         i_train = [i for i in range(trainingsize)]
    #         rapp = apprentice.RationalApproximation(X[i_train],Y[i_train],order=(m,n), strategy=1)
    #
    #         # rappsip  = apprentice.RationalApproximationSIP(X[i_train],Y[i_train],m=m,n=n,trainingscale="Cp",
    #         #                     strategy=0,roboptstrategy = 'msbarontime',fitstrategy = 'filter',localoptsolver = 'scipy')
    #         # rappsip.save(folder+"/rappsip.json")
    #
    #
    #         rapp.save(folder+"/rapp.json")
    #
    #         testfile = "../benchmarkdata/"+name+"_test.txt"
    #
    #         plot2Dsurface(folder+"/rapp.json", testfile, folder, name+noisestr+"_rapp","all")
    #
    #         rappsipfile = "%s%s_2x/out/%s%s_2x_p%d_q%d_ts2x.json"%(name,noisestr,name,noisestr,m,n)
    #         plot2Dsurface(rappsipfile, testfile, folder, name+noisestr+"_rappsip","all")
    #
    #         # plot2Dsurface(folder+"/rappsip.json", testfile, folder, name+noisestr+"_rappsip","all")
    #
    # exit(1)





    import os, sys
    if len(sys.argv)!=8:
        print("Usage: {} fno noise m n ts fndesc papp_or_no_papp".format(sys.argv[0]))
        sys.exit(1)

    # noisearr = sys.argv[2].split(',')
    # if len(noisearr) == 0:
    #     print("please specify comma saperated noise levels")
    #     sys.exit(1)

    # if not os.path.exists(sys.argv[4]):
    #     print("Test file '{}' not found.".format(sys.argv[4]))
    #     exit(1)

    plot2Dsurface(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7])
###########
