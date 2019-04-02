import numpy as np
from apprentice import RationalApproximationSIP, PolynomialApproximation
from apprentice import tools, readData
import os
from mpl_toolkits.mplot3d import Axes3D
import os, sys

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

def plotoptiterationmaps(farr,noisearr, ts):
    import glob
    import json
    import re
    import json
    openfileStr = ""
    lx="$\\log_{10}(\\alpha(M) + \\alpha(N))$"
    ly="$\\log_{10}(\\Delta_{MN})$"
    logy=True
    logx=True
    for num, fname in enumerate(farr):
        box = getbox(fname)
        if(len(box) != 2):
            print("{} cannot handle dim != 2. Box len was {}".format(sys.argv[0],len(box)))
            sys.exit(1)
        npoints = 1000
        X_test1 = np.linspace(box[0][0], box[0][1], num=npoints)
        X_test2 = np.linspace(box[1][0], box[1][1], num=npoints)

        for noise in noisearr:
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)
            if not os.path.exists(folder):
                print("Folder '{}' not found.".format(folder))
                sys.exit(1)
            if not os.path.exists(folder+"/plots"):
                os.mkdir(folder+'/plots')
            filelist = np.array(glob.glob(folder+"/out/*.json"))
            filelist = np.sort(filelist)

            for file in filelist:
                if file:
                    with open(file, 'r') as fn:
                        datastore = json.load(fn)
                if(datastore["dim"] != 2):
                    print("{} cannot handle dim != 2 Dim found in datastore was {}".format(sys.argv[0],datastore["dim"]))
                    sys.exit(1)
                if(len(datastore["iterationinfo"]) != 3):
                    continue
                # if(datastore["m"] !=5 or datastore["n"] !=3):
                #     continue
                print(file)
                maxy_pq = 0
                miny_pq = np.inf
                maxy_q = 0
                miny_q = np.inf
                data = {}
                m = datastore["m"]
                n = datastore["n"]

                outx1 = "%s/plots/Cimap_X_%s%s_p%d_q%d_ts%s.csv"%(folder, fname, noisestr,m,n,ts)
                outy_pq = "%s/plots/Cimap_Y_pq_%s%s_p%d_q%d_ts%s.csv"%(folder, fname, noisestr,m,n,ts)
                outy_q = "%s/plots/Cimap_Y_q_%s%s_p%d_q%d_ts%s.csv"%(folder, fname, noisestr,m,n,ts)
                polex  = "%s/plots/Cimap_pole_x_%s%s_p%d_q%d_ts%s.csv"%(folder, fname, noisestr,m,n,ts)

                for iterno in range(3):
                    data[iterno] = {}
                    if file:
                        with open(file, 'r') as fn:
                            datastore = json.load(fn)
                    ii = datastore["iterationinfo"]
                    datastore['pcoeff'] = ii[iterno]['pcoeff']
                    datastore['qcoeff'] = ii[iterno]['qcoeff']
                    iro = ii[iterno]['robOptInfo']['robustArg']
                    iro.append(ii[iterno]['robOptInfo']['robustObj'])
                    data[iterno]['robarg'] = iro

                    rappsip = RationalApproximationSIP(datastore)
                    Y_pred_pq = []
                    for x in X_test1:
                        for y in X_test2:
                            Y_pred_pq.append(rappsip.predict([x,y]))
                    mmm = max(Y_pred_pq)
                    # print(mmm)
                    if(mmm > maxy_pq):
                        maxy_pq = mmm
                    mmm = min(Y_pred_pq)
                    # print(mmm)
                    if(mmm < miny_pq):
                        miny_pq = mmm
                    data[iterno]['Y_pred_pq'] = Y_pred_pq

                    Y_pred_q = []
                    for x in X_test1:
                        for y in X_test2:
                            X = [x,y]
                            X = rappsip._scaler.scale(np.array(X))
                            Y_pred_q.append(rappsip.denom(X))
                    mmm = max(Y_pred_q)
                    # print(mmm)
                    if(mmm > maxy_q):
                        maxy_q = mmm
                    mmm = min(Y_pred_q)
                    # print(mmm)
                    if(mmm < miny_q):
                        miny_q = mmm
                    data[iterno]['Y_pred_q'] = Y_pred_q

                    np.savetxt(outx1,np.stack((X_test1,X_test2),axis=1), delimiter=",")

                np.savetxt(outy_pq,np.stack((data[0]['Y_pred_pq'],data[1]['Y_pred_pq'],data[2]['Y_pred_pq']),axis=1), delimiter=",")
                np.savetxt(outy_q,np.stack((data[0]['Y_pred_q'],data[1]['Y_pred_q'],data[2]['Y_pred_q']),axis=1), delimiter=",")
                np.savetxt(polex,np.stack((data[0]['robarg'],data[1]['robarg'],data[2]['robarg']),axis=1), delimiter=",")

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
                x1lim = (box[0][0],box[0][1])
                x2lim = (box[1][0],box[1][1])
                fig = plt.figure(figsize=(17,8))
                ypredmaster = []
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

                # axarray = np.array([])
                for iterno in range(3):
                    ax = fig.add_subplot(2, 3, iterno+1, projection='3d')
                    min111 = min(data[iterno]['Y_pred_pq'])
                    max111 = max(data[iterno]['Y_pred_pq'])
                    Y_pred_pq = np.reshape(data[iterno]['Y_pred_pq'], [len(X_test1), len(X_test2)])
                    # im1 = ax.contour3D(X_test1, X_test2, Y_pred_pq, 100, cmap=cmap2, norm = colors.SymLogNorm(linthresh=1, linscale=1,vmin=miny_pq, vmax=maxy_pq),alpha=0.8)
                    im1 = ax.contour3D(X_test1, X_test2, Y_pred_pq, 100, cmap=cmap, norm = colors.SymLogNorm(linthresh=0.2, linscale=0.5,vmin=miny_pq, vmax=maxy_pq),alpha=0.8)

                    # im1 = ax.contour3D(X_test1, X_test2, Y_pred_pq, cmap=matplotlib.cm.seismic,vmin=miny_pq, vmax=maxy_pq)
                    # axarray = np.append(axarray,ax)
                    if(iterno == 0):
                        ypredmaster = Y_pred_pq
                    if(iterno == 2):
                        mmm = plt.cm.ScalarMappable(cmap=cmap)
                        mmm.set_array(ypredmaster)
                        mmm.set_clim(miny_pq, maxy_pq)
                        b1=fig.colorbar(mmm)
                        b1.set_label("$\\frac{p(x_1,x_2)}{q(x_1,x_2)}$", fontsize = 16)

                    # ax.set_title('Iteration: %d'%(iterno+1), fontsize = 12)
                    ax.set_xlabel('$x_1$', fontsize = 14)
                    ax.set_ylabel('$x_2$', fontsize = 14)
                    # b1 = fig.colorbar(im1,ax=ax, shrink=0.95,extend='both')
                    # if(iterno ==2):
                    #     b1.set_label("$\\frac{p_m(x_1,x_2)}{q_n(x_1,x_2)}$", fontsize = 14)
                    ax.set(xlim=x1lim,ylim=x2lim)
                    ax.view_init(60, 35)
                    if(iterno !=2):
                        ax.scatter(data[iterno]['robarg'][1], data[iterno]['robarg'][0], marker = '*', c = "black"  ,s=333, alpha = 1,zorder=1)



                for iterno in range(3):
                    ax = fig.add_subplot(2, 3, iterno+4, projection='3d')
                    min111 = min(data[iterno]['Y_pred_pq'])
                    max111 = max(data[iterno]['Y_pred_pq'])
                    Y_pred_q = np.reshape(data[iterno]['Y_pred_q'], [len(X_test1), len(X_test2)])
                    im2 = ax.contour3D(X_test1, X_test2, Y_pred_q,cmap=cmap, norm = colors.SymLogNorm(linthresh=4, linscale=0.6,vmin=miny_q, vmax=maxy_q))
                    # im2 = ax.contour3D(X_test1, X_test2, Y_pred_q, cmap=cmap2,vmin=miny_q, vmax=maxy_q)
                    # ax.set_title('Iteration: %d'%(iterno+1), fontsize = 12)
                    ax.set_xlabel('$x_1$', fontsize = 14)
                    ax.set_ylabel('$x_2$', fontsize = 14)
                    # b2 = fig.colorbar(im2,ax=ax, shrink=0.95,extend='both')
                    if(iterno == 0):
                        ypredmaster = Y_pred_q
                    if(iterno == 2):
                        mmm = plt.cm.ScalarMappable(cmap=cmap)
                        mmm.set_array(ypredmaster)
                        mmm.set_clim(miny_q, maxy_q)
                        b2=fig.colorbar(mmm)
                        b2.set_label("$q(x_1,x_2)$", fontsize = 12)

                    ax.view_init(60, 35)
                    if(iterno !=2):
                        ax.scatter(data[iterno]['robarg'][1], data[iterno]['robarg'][0],  marker = '*', c = "black"  ,s=333, alpha = 1,zorder=1)
                # b2=fig.colorbar(im2,ax=ax, shrink=0.95,extend='both')
                # fig.suptitle("%s%s. m = %d, n = %d. trainingsize = %s "%(fname, noisestr,m,n,ts), fontsize = 18)
                # plt.show()


                # exit(1)
                #
                # cmap2='hot'
                # import matplotlib
                # import matplotlib.colors as colors
                # cmap1 = matplotlib.cm.RdYlBu
                # import matplotlib as mpl
                # import matplotlib.pyplot as plt
                # import matplotlib.text as text
                # fig, axarr = plt.subplots(2,3, figsize=(17,8))
                # x1lim = (box[0][0]-0.5,box[0][1]+0.5)
                # x2lim = (box[1][0]-0.5,box[1][1]+0.5)
                # # if(abs(miny_pq) <  abs(maxy_pq)):
                # #     miny_pq = -abs(maxy_pq)
                # #     maxy_pq = abs(maxy_pq)
                # # elif(abs(miny_pq) >  abs(maxy_pq)):
                # #     miny_pq = -abs(miny_pq)
                # #     maxy_pq = abs(miny_pq)
                # # xx,yy = np.meshgrid(X_test1,X_test2)
                # for iterno in range(3):
                #     min111 = min(data[iterno]['Y_pred_pq'])
                #     max111 = max(data[iterno]['Y_pred_pq'])
                #     # print(data[iterno]['Y_pred'])
                #     Y_pred_pq = np.reshape(data[iterno]['Y_pred_pq'], [len(X_test1), len(X_test2)])
                #     # print(Y_pred_pq)
                #     # im1 = axarr[0][iterno].pcolormesh(X_test1, X_test2, Y_pred_pq, cmap=cmap, vmin=min111, vmax=max111)
                #     # im1 = axarr[0][iterno].pcolormesh(X_test1,X_test2, Y_pred_pq, cmap=cmap)
                #     # im1 = axarr[0][iterno].pcolormesh(X_test1, X_test2, Y_pred_pq)
                #     # im1 = axarr[0][iterno].pcolormesh(X_test1, X_test2, Y_pred_pq, cmap=cmap1, vmin=miny_pq, vmax=maxy_pq)
                #     im1 = axarr[0][iterno].pcolormesh(X_test1, X_test2, Y_pred_pq, cmap=cmap1, norm = colors.SymLogNorm(linthresh=0.1, linscale=1,vmin=miny_pq, vmax=maxy_pq))
                #     # im1 = axarr[0][iterno].imshow(Y_pred_pq, extent=[-1, 1, -1, 1], cmap=cmap1)
                #     # axarr[0][iterno].axis(aspect='image')
                #     axarr[0][iterno].set_xlabel('$x_1$', fontsize = 14)
                #     axarr[0][iterno].set_ylabel('$x_2$', fontsize = 14)
                #     axarr[0][iterno].set(xlim=x1lim,ylim=x2lim)
                #     if(iterno !=2):
                #         axarr[0][iterno].scatter(data[iterno]['robarg'][1], data[iterno]['robarg'][0], marker = 'x', c = "black"  ,s=50, alpha = 1)
                #     # if(iterno ==0):
                #     # fig.colorbar(im1)
                #     # axarr[0][iterno].axis('tight')
                #
                #     Y_pred_q = np.reshape(data[iterno]['Y_pred_q'], [len(X_test1), len(X_test2)])
                #     # im2 = axarr[1][iterno].contour(X_test1, X_test2, Y_pred_q, cmap=cmap2, vmin=miny_q, vmax=maxy_q)
                #     im2 = axarr[1][iterno].pcolormesh(X_test1, X_test2, Y_pred_q, cmap=cmap1, norm = colors.SymLogNorm(linthresh=4, linscale=0.6,vmin=miny_q, vmax=maxy_q))
                #     axarr[1][iterno].set_xlabel('$x_1$', fontsize = 14)
                #     axarr[1][iterno].set_ylabel('$x_2$', fontsize = 14)
                #     axarr[1][iterno].set(xlim=x1lim,ylim=x2lim)
                #
                #     if(iterno !=2):
                #         axarr[1][iterno].scatter(data[iterno]['robarg'][1], data[iterno]['robarg'][0], marker = 'x', c = "black"  ,s=50, alpha = 1)
                # for ax in axarr.flat:
                #     ax.label_outer()
                # b1=fig.colorbar(im1,ax=axarr[0].ravel().tolist(), shrink=0.95,extend='both')
                # b1.set_label("$\\frac{p_m(x_1,x_2)}{q_n(x_1,x_2)}$", fontsize = 16)
                # b2=fig.colorbar(im2,ax=axarr[1].ravel().tolist(), shrink=0.95,extend='both')
                # b2.set_label("$q_n(x_1,x_2)$", fontsize = 16)
                #
                # fig.suptitle("%s%s. m = %d, n = %d. trainingsize = %s "%(fname, noisestr,m,n,ts), fontsize = 18)

                if not os.path.exists(folder+"/plots"):
                    os.mkdir(folder+'/plots')

                outfile = "%s/plots/Pimap_%s%s_p%d_q%d_ts%s.pdf"%(folder, fname, noisestr,m,n,ts)
                plt.savefig(outfile)
                plt.clf()
                openfileStr += "open "+outfile+"; "
                print(openfileStr)
        plt.close('all')
    print(openfileStr)








if __name__ == "__main__":

    if len(sys.argv)!=4:
        print("Usage: {} functions noiseToConsider ts".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise.")
        sys.exit(1)


    plotoptiterationmaps(farr,noisearr, sys.argv[3])
###########
