
import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
import scipy
import matplotlib.ticker as mtick
import os

def knowmissing(filename):
    arr = [

    ]
    for a in arr:
        if(filename == a):
            return 1
    return 0

def getdim(fname):
    dim = {"f1":2,"f2":2,"f3":2,"f4":2,"f5":2,"f7":2,"f8":2,"f9":2,"f10":4,"f12":2,"f13":2,
            "f14":2,"f15":2,"f16":2,"f17":3,"f18":4,"f19":4,"f20":4,"f21":2,"f22":2}
    return dim[fname]

def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]


def getstats(data,metr):
    def geomean(iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    if metr == 'amean': return np.average(data)
    elif metr == 'gmean': return geomean(data)
    elif metr == 'median': return np.median(data)
    elif metr == 'range': return np.max(data) - np.min(data)
    else:
        print("metric not known")
        exit(1)




def tabletotalcputime(farr,noisearr, ts, table_or_latex):
    print (farr)
    print (noisearr)

    # allsamples = ['mc','lhs','so','sg']
    # allsamples = ['lhs','splitlhs','sg']
    # allsamples = ['sg']
    # allsamples = ['splitlhs']
    # allsamples = ['lhs','splitlhs']
    allsamples = ['sg','lhs','splitlhs']
    allsampleslabels = ['SG','LHS','d-LHD']
    import json
    from apprentice import tools
    results = {}
    dumpr = {}
    for snum, sample in enumerate(allsamples):
        results[sample] = {}
        dumpr[sample] = {}
        for num,fname in enumerate(farr):
            results[sample][fname] = {}
            m = 5
            n = 5

            for noise in noisearr:
                results[sample][fname][noise] = {}
                noisestr,noisepct = getnoiseinfo(noise)

                timepa = []
                timera = []
                timerard = []
                timerasip = []
                iterrasip = []
                iterrasipnonlog = []
                fittime = []
                mstime = []
                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    # print(folder)
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    pappfile = "%s/outpa/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)

                    if not os.path.exists(rappsipfile):
                        print("rappsipfile %s not found"%(rappsipfile))
                        if(knowmissing(rappsipfile)):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(rappfile):
                        print("rappfile %s not found"%(rappfile))
                        if(knowmissing(rappfile)):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(rapprdfile):
                        print("rapprdfile %s not found"%(rapprdfile))
                        if(knowmissing(rapprdfile)):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(pappfile):
                        print("pappfile %s not found"%(pappfile))
                        if(knowmissing(pappfile)):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    dim = getdim(fname)
                    rdof = tools.numCoeffsRapp(dim,[int(m),int(n)])
                    rpnnl = tools.numCoeffsPoly(dim,m) - (dim+1)
                    rqnnl = tools.numCoeffsPoly(dim,n) - (dim+1)

                    if rappsipfile:
                        with open(rappsipfile, 'r') as fn:
                            datastore = json.load(fn)
                    rappsiptime = datastore['log']['fittime']
                    rnoiters = len(datastore['iterationinfo'])
                    timerasip.append(rappsiptime)
                    # timerasip.append(rappsiptime)
                    iterrasip.append(rnoiters)
                    iterrasipnonlog.append(rnoiters)
                    ft = 0.
                    mt = 0.
                    for iter in datastore['iterationinfo']:
                        ft+=iter['log']['time']
                        mt+=iter['robOptInfo']['info'][0]['log']['time']
                    fittime.append(ft)
                    mstime.append(mt)
                    # fittime.append(ft)
                    # mstime.append(mt)


                    if rappfile:
                        with open(rappfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapptime = datastore['log']['fittime']
                    timera.append(rapptime)
                    # timera.append(rapptime)

                    if rapprdfile:
                        with open(rapprdfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapprdtime = datastore['log']['fittime']
                    timerard.append(rapprdtime)
                    # timerard.append(rapprdtime)


                    if pappfile:
                        with open(pappfile, 'r') as fn:
                            datastore = json.load(fn)
                    papptime = datastore['log']['fittime']
                    pdof = tools.numCoeffsPoly(datastore['dim'],datastore['m'])
                    timepa.append(papptime)
                    # timepa.append(papptime)
                    if(sample == "sg"):
                        break


                missingmean = -1

                if len(timerard) == 0:
                    rapprd = missingmean
                    rapprdsd = 0
                else:
                    rapprd = np.average(timerard)
                    rapprdsd = np.std(timerard)


                if len(timera) == 0:
                    rapp = missingmean
                    rappsd = 0
                else:
                    rapp = np.average(timera)
                    rappsd = np.std(timera)


                if len(timepa) == 0:
                    papp = missingmean
                    pappsd = 0
                else:
                    papp = np.average(timepa)
                    pappsd = np.std(timepa)


                if len(timerasip) == 0:
                    rappsip = missingmean
                    rappsipsd = 0
                else:
                    rappsip = np.average(timerasip)
                    rappsipsd = np.std(timerasip)


                if len(iterrasip) == 0:
                    rnoiters = missingmean
                    rnoiterssd = 0
                else:
                    rnoiters = np.average(iterrasip)
                    rnoiterssd = np.std(iterrasip)

                if len(fittime) == 0:
                    rfittime = missingmean
                    rfittimesd = 0
                else:
                    rfittime = np.average(fittime)
                    rfittimesd = np.std(fittime)

                if len(mstime) == 0:
                    rmstime= missingmean
                    rmstimesd = 0
                else:
                    rmstime = np.average(mstime)
                    rmstimesd = np.std(mstime)



                results[sample][fname][noise] = {
                    "rapprd":rapprd,
                    "rapprdsd":rapprdsd,

                    "rapp":rapp,
                    "rappsd":rappsd,

                    "rappsip":rappsip,
                    "rappsipsd":rappsipsd,

                    "papp":papp,
                    "pappsd":pappsd,

                    'rnoiters':rnoiters,
                    'rnoiterssd':rnoiterssd,

                    'rfittime':rfittime,
                    'rfittimesd':rfittimesd,

                    'rmstime':rmstime,
                    'rmstimesd':rmstimesd,

                    'pdof':pdof,
                    'rdof':rdof,
                    'rpnnl':rpnnl,
                    'rqnnl':rqnnl
                }
                dumpr[sample][fname] = iterrasipnonlog



        # from IPython import embed
        # embed()

    # print(results)

    #############################################
    #iteration summary latex
    #############################################
    # python tabletotalcputime.py  f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0 2x latex
    noise = noisearr[0]
    metricarr = ['amean','gmean','median','range']
    metricarrlabel = ['Arithmetic Mean','Geometric Mean','Median','Range']
    stats = {}
    s = ""
    for mnum, metr in enumerate(metricarr):
        s+="%s"%(metricarrlabel[mnum])
        for snum,sample in enumerate(allsamples):
            data = []
            for fnum,fname in enumerate(farr):
                data.append(results[sample][fname][noise]['rnoiters'])
                if(fname == 'f20'):
                    print(results[sample][fname][noise]['rnoiters'])

            print(np.max(data),np.min(data))
            stat = getstats(data,metr)
            s+="&%.2f"%(stat)
        s+="\n\\\\\\hline\n"
    print(s)




    #############################################
    #cputime summary latex
    #############################################
    noise = noisearr[0]
    metricarr = ['amean','gmean','median','range']
    metricarrlabel = ['Arithmetic Mean','Geometric Mean','Median','Range']
    methodarr = ['papp','rapp','rapprd','rfittime','rmstime']
    stats = {}
    sample = 'splitlhs'
    s = ""


    for mnum, metr in enumerate(metricarr):
        s+="%s"%(metricarrlabel[mnum])
        for menum,method in enumerate(methodarr):
            data = []
            for fnum,fname in enumerate(farr):
                data.append(results[sample][fname][noise][method])


            stat = getstats(data,metr)
            s+="&%.2f"%(stat)
        s+="\n\\\\\\hline\n"
    print(s)

    #############################################
    #cputime and iteration electronic suppliment latex
    #############################################
    # python tabletotalcputime.py  f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0 2x latex
    # python tabletotalcputime.py  f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 10-6 2x latex
    # python tabletotalcputime.py  f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 10-2 2x latex
    noise = noisearr[0]
    s = ""
    keyarr = ['rapp','rapprd','rfittime','rmstime','rnoiters','papp']

    for fnum, fname in enumerate(farr):
        s += "\\multirow{3}{*}{\\ref{fn:%s}}"%(fname)
        for snum, sample in enumerate(allsamples):
            s+="&%s"%(allsampleslabels[snum])
            for knum, key in enumerate(keyarr):
                statsarr = [key,key+'sd']
                for stnum, stat in enumerate(statsarr):
                    val = results[sample][fname][noise][stat]
                    if sample == 'sg' and stnum == 1:
                        s+="&-"
                    elif val == int(val):
                        s+="&%d"%(int(val))
                    elif val < 10**-2 or val >10**2:
                        s+="&%.2E"%(val)
                    else:
                        s+="&%.2f"%(val)
            if snum < len(allsamples)-1:
                s+="\n\\\\*  \\cline{2-14}\n"
        s+="\n\\\\ \\hline\n"
    print(s)






    exit(1)
    import json
    with open("results/plots/Jiterations.json", "w") as f:
            json.dump(dumpr, f,indent=4, sort_keys=True)

    baseline = 2
    totalrow = 3
    totalcol = 3
    import matplotlib.pyplot as plt
    ffffff = plt.figure(0,figsize=(45, 20))
    axarray = []
    width = 0.21
    ecolor = 'black'
    X111 = np.arange(len(farr))
    meankeyarr = ['papp','rapp','rapprd','rappsip']
    sdkeyarr = ['pappsd','rappsd','rapprdsd','rappsipsd']
    legendarr = ['Polynomial Approx. ','Algorithm \\ref{ALG:MVVandQR} without degree reduction','Algorithm \\ref{ALG:MVVandQR}' ,'MST Algorithm \\ref{A:Polyak}']
    legend2arr = [None,None,None,'FT Algorithm \\ref{A:Polyak}']
    color = ['#FFC300','#FF5733','#C70039','#900C3F']
    colorfittime = ['yellow','yellow','yellow','yellow']
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    for nnum,noise in enumerate(noisearr):
        for snum, sample in enumerate(allsamples):
            mean ={}
            fitmean = {}
            for type in meankeyarr:
                mean[type] = []
                fitmean[type] = []
            sd = {}
            fitsd = {}
            for type in sdkeyarr:
                sd[type] = []
                fitsd[type] = []
            for fname in farr:
                for type in meankeyarr:
                    mean[type].append(results[sample][fname][noise][type])
                    # mean[type].append(np.ma.log10(results[sample][fname][noise][type]))
                for type in sdkeyarr:
                    # print(results[sample][fname][noise][type])
                    sd[type].append(results[sample][fname][noise][type])
                    # sd[type].append(np.ma.log10(results[sample][fname][noise][type]))

                for type in meankeyarr:
                    if(type == "rappsip"):
                        fitmean[type].append(results[sample][fname][noise]['rfittime'])
                    else:
                        fitmean[type].append(-1*baseline)
                for type in sdkeyarr:
                    if(type == "rappsip"):
                        fitsd[type].append(results[sample][fname][noise]['rfittimesd'])
                    else:
                        fitsd[type].append(0)

            if(len(axarray)>0):
                ax = plt.subplot2grid((totalrow,totalcol), (nnum,snum),sharex=axarray[0],sharey=axarray[0])
                axarray.append(ax)
            else:
                ax = plt.subplot2grid((totalrow,totalcol), (nnum,snum))
                axarray.append(ax)

            # print(mean)
            # print(sd)
            for typenum,type in enumerate(meankeyarr):
                sdkey = sdkeyarr[typenum]
                # print(mean[type])
                if(sample == 'sg'):
                    ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], capsize=3,label=legendarr[typenum])
                else:
                    ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], yerr=np.array(sd[sdkey]),align='center',  ecolor=ecolor, capsize=3)
            for typenum,type in enumerate(meankeyarr):
                sdkey = sdkeyarr[typenum]
                if(sample == 'sg'):
                    ax.bar(X111+typenum*width, np.array(fitmean[type])+baseline, width,color=colorfittime[typenum], capsize=3,label=legend2arr[typenum])
                else:
                    ax.bar(X111+typenum*width, np.array(fitmean[type])+baseline, width,color=colorfittime[typenum], yerr=np.array(fitsd[sdkey]),align='center',  ecolor=ecolor, capsize=3)
                ax.set_xticks(X111 + (len(meankeyarr)-1)*width / 2)
                xlab = []
                for f in farr:
                    # print(f)
                    # xlab.append("\\ref{fn:%s}"%(f))
                    # xlab.append("\\ref{%s}"%(f))
                    xlab.append("%s"%(f))
                ax.set_xticklabels(xlab,fontsize = 20)
                ax.set_xlabel("Test functions",fontsize=22)
                ax.set_ylabel("$\\log_{10}$ [CPU time (sec)]",fontsize=22)
                ax.label_outer()

        if(nnum==0):
            l1 = ffffff.legend(
                loc='upper center', ncol=5,fontsize = 25,borderaxespad=0.,shadow=False)


    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))

    ffffff.savefig("../../log/cputime.png", bbox_extra_artists=(l1,), bbox_inches='tight')

    plt.clf()
    plt.close('all')

    # Filtered plot
    totalrow = 3
    totalcol = 1
    import matplotlib.pyplot as plt
    ffffff = plt.figure(0,figsize=(45, 20))
    X111 = np.arange(len(farr)*len(noisearr))
    axarray = []
    width = 0.15
    ecolor = 'black'
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    for snum, sample in enumerate(allsamples):
        mean ={}
        fitmean = {}
        for type in meankeyarr:
            mean[type] = []
            fitmean[type] = []
        sd = {}
        fitsd = {}
        for type in sdkeyarr:
            sd[type] = []
            fitsd[type] = []
        for nnum,noise in enumerate(noisearr):
            for fname in farr:
                for type in meankeyarr:
                    mean[type].append(results[sample][fname][noise][type])
                    # mean[type].append(np.ma.log10(results[sample][fname][noise][type]))
                for type in sdkeyarr:
                    # print(results[sample][fname][noise][type])
                    sd[type].append(results[sample][fname][noise][type])
                    # sd[type].append(np.ma.log10(results[sample][fname][noise][type]))

                for type in meankeyarr:
                    if(type == "rappsip"):
                        fitmean[type].append(results[sample][fname][noise]['rfittime'])
                    else:
                        fitmean[type].append(-1*baseline)
                for type in sdkeyarr:
                    if(type == "rappsip"):
                        fitsd[type].append(results[sample][fname][noise]['rfittimesd'])
                    else:
                        fitsd[type].append(0)

        if(len(axarray)>0):
            ax = plt.subplot2grid((totalrow,totalcol), (snum,0),sharex=axarray[0],sharey=axarray[0])
            axarray.append(ax)
        else:
            ax = plt.subplot2grid((totalrow,totalcol), (snum,0))
            axarray.append(ax)

        ax.set_xlim(-.3,14.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.axvspan(-.3, 3.7, alpha=0.5, color='pink')
        plt.axvspan(3.7, 9.7, alpha=0.5, color='lightgrey')
        plt.axvspan(9.7, 14.7, alpha=0.5, color='cyan')
        for typenum,type in enumerate(meankeyarr):
            sdkey = sdkeyarr[typenum]
            # print(mean[type])
            if(sample == 'sg'):
                ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], capsize=3,label=legendarr[typenum])
            else:
                ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], yerr=np.array(sd[sdkey]),align='center',  ecolor=ecolor, capsize=3,label=legendarr[typenum])
        for typenum,type in enumerate(meankeyarr):
            sdkey = sdkeyarr[typenum]
            if(sample == 'sg'):
                ax.bar(X111+typenum*width, np.array(fitmean[type])+baseline, width,color=colorfittime[typenum], capsize=3,label=legend2arr[typenum])
            else:
                ax.bar(X111+typenum*width, np.array(fitmean[type])+baseline, width,color=colorfittime[typenum], yerr=np.array(fitsd[sdkey]),align='center',  ecolor=ecolor, capsize=3,label=legend2arr[typenum])

        if(snum==0):
            l1 = ffffff.legend(
                loc='upper center', ncol=5,fontsize = 25)
            noiselegendarr = ['$\\epsilon=0$','$\\epsilon=10^{-6}$','$\\epsilon=10^{-2}$']
            l2 = ffffff.legend(noiselegendarr,loc='upper center', ncol=4,bbox_to_anchor=(0.435, 0.85), fontsize = 20,borderaxespad=0.,shadow=False)
        ax.set_xticks(X111 + (len(meankeyarr)-1)*width / 2)
        xlab = []
        for f in farr:
            # print(f)
            # xlab.append("\\ref{fn:%s}"%(f))
            # xlab.append("\\ref{%s}"%(f))
            xlab.append("%s"%(f))
        xlab1 = np.concatenate((xlab,xlab,xlab),axis=None)
        ax.set_xticklabels(xlab1,fontsize = 20)
        ax.set_xlabel("Test functions",fontsize=22)
        ax.set_ylabel("$\\log_{10}$ [CPU time (sec)]",fontsize=22)
        ax.label_outer()




    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))

    ffffff.savefig("../../log/cputime2.png", bbox_extra_artists=(l1,), bbox_inches='tight')

    plt.clf()
    plt.close('all')


    # CPU time plot for paper
    import matplotlib as mpl
    # mpl.use('pgf')
    # pgf_with_custom_preamble = {
    #     "text.usetex": True,    # use inline math for ticks
    #     "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    #     "pgf.preamble": [
    #         "\\usepackage{amsmath}",         # load additional packages
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_custom_preamble)




    totalrow = 1
    totalcol = 1
    meankeyarr1 = ['papp','rapp','rapprd','rappsip','rfittime']
    sdkeyarr1 = ['pappsd','rappsd','rapprdsd','rappsipsd','rmstimesd']
    # color1 = ['#900C3F','#C70039','#FF5733','#FFC300','pink']
    color1 = ["m", "c", "g", "b"]
    # legendarr1 = ['Polynomial Approximation ','Algorithm \\ref{ALG:MVVandQR} without degree reduction','Algorithm \\ref{ALG:MVVandQR} with degree reduction' ,'Algorithm \\ref{A:Polyak}: fit time','Algorithm \\ref{A:Polyak}: multistart time']
    legendarr1 = ['$p(x)$','$r_1(x)$','$r_2(x)$' ,'$r_3(x)\\mathrm{:\\ multistart\\ time}$','$r_3(x)\\mathrm{:\\ fit\\ time}$']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.rc('font', weight='bold')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    # mpl.style.use("ggplot")


    mpl.rc('font',family='serif')

    ffffff = plt.figure(0,figsize=(15, 10))
    X111 = np.arange(len(farr)*len(noisearr))
    axarray = []
    width = 0.2
    ecolor = 'black'
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    for snum, sample in enumerate(allsamples):
        mean = {}
        for type in meankeyarr1:
            mean[type] = []
        sd = {}
        for type in sdkeyarr1:
            sd[type] = []
        for nnum,noise in enumerate(noisearr):
            for fname in farr:
                for type in meankeyarr1:
                    mean[type].append(results[sample][fname][noise][type])
                    # mean[type].append(np.ma.log10(results[sample][fname][noise][type]))
                for type in sdkeyarr1:
                    # print(results[sample][fname][noise][type])
                    sd[type].append(results[sample][fname][noise][type])
                    # sd[type].append(np.ma.log10(results[sample][fname][noise][type]))

        if(len(axarray)>0):
            ax = plt.subplot2grid((totalrow,totalcol), (snum,0),sharex=axarray[0],sharey=axarray[0])
            axarray.append(ax)
        else:
            ax = plt.subplot2grid((totalrow,totalcol), (snum,0))
            axarray.append(ax)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlim(-.3,14.7)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.axvspan(-.3, 3.7, alpha=0.5, color='pink')
        # plt.axvspan(3.7, 9.7, alpha=0.5, color='lightgrey')
        # plt.axvspan(9.7, 14.7, alpha=0.5, color='cyan')
        alignarr = [0.5,0.5,0.5,0.2]
        for typenum,type in enumerate(meankeyarr1):
            sdkey = sdkeyarr1[typenum]
            # print(mean[type])
            if(sample == 'sg'):
                ax.bar(X111+typenum*width, np.array(mean[type]), width,color=color1[typenum], capsize=3,label=legendarr1[typenum])
            else:
                ax.bar(X111+typenum*width, np.array(mean[type]), width,color=color1[typenum], alpha=alignarr[typenum],align='center',  ecolor=ecolor, capsize=3,label=legendarr1[typenum])
                ax.vlines(X111+typenum*width, np.array(mean[type]),np.array(mean[type])+np.array(sd[sdkey]))
            if(typenum == 3):
                newtn = typenum + 1
                newtype = meankeyarr1[newtn]
                newsdkey = sdkeyarr1[newtn]
                ax.bar(X111+typenum*width, np.array(mean[newtype]), width,color=color1[typenum], alpha=0.5,align='center',  ecolor=ecolor, capsize=3,label=legendarr1[newtn])
                ax.vlines(X111+typenum*width, np.array(mean[type]),np.array(mean[type])+np.array(sd[newsdkey]))
                break


        if(snum==0):
            l1 = ax.legend(
                loc='upper left', ncol=1,fontsize = 20,framealpha=1,shadow=True,frameon=False)
            l1.get_frame().set_facecolor('white')
            noiselegendarr = ['$\\epsilon=0$','$\\epsilon=10^{-6}$','$\\epsilon=10^{-2}$']
            # l2 = ffffff.legend(noiselegendarr,loc='upper center', ncol=4,bbox_to_anchor=(0.435, 0.85), fontsize = 20,borderaxespad=0.,shadow=False)
        ax.set_xticks(X111 + (len(meankeyarr)-1)*width / 2)
        ax.set_yscale("log")
        xlab = []
        for f in farr:
            # print(f)
            xlab.append("\\ref{fn:%s}"%(f))
            # xlab.append("\\ref{%s}"%(f))
            # xlab.append("%s"%(f))
        # xlab1 = np.concatenate((xlab,xlab,xlab),axis=None)
        xlab11= ['$A.1.4$','$A.1.7$','$A.1.15$','$A.1.16$','$A.1.17$']
        ax.set_xticklabels(xlab11,fontsize = 20)

        plt.rc('ytick',labelsize=20)
        plt.rc('xtick',labelsize=20)
        plt.tick_params(labelsize=20)
        # ax.set_xlabel("Test functions",fontsize=22)
        ax.set_ylabel("$\\mathrm{CPU\\ time\\ (sec)}$",fontsize=20)
        # for axis in [ax.xaxis, ax.yaxis]:
        #     axis.set_major_formatter(ScalarFormatter())


        ax.label_outer()






    # plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))

    # ffffff.savefig("../../log/cputimeplot.pgf", bbox_extra_artists=(l1,), bbox_inches='tight')
    # ffffff.savefig("../../log/cputimeplot.png", bbox_extra_artists=(l1,), bbox_inches='tight')
    ffffff.savefig("../../log/cputimeplot.pdf", bbox_extra_artists=(l1,), bbox_inches='tight')

    plt.clf()
    plt.close('all')

    exit(1)








    # Iteration plot
    import matplotlib as mpl
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.rc('font', weight='bold')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    # mpl.style.use("ggplot")
    # mpl.use('pgf')
    # pgf_with_custom_preamble = {
    #     "text.usetex": True,    # use inline math for ticks
    #     "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    #     "pgf.preamble": [
    #         "\\usepackage{amsmath}",         # load additional packages
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_custom_preamble)
    color = ['#FFC300','#FF5733','#900C3F']
    X111 = np.arange(len(farr))
    ffffff = plt.figure(0,figsize=(15, 10))
    # ffffff = plt.figure(0)
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    totalrow = 1
    totalcol = len(noisearr)
    baseline = 0
    width = 0.4
    axarray = []
    legendarr = ['$\\mathrm{Latin\\ Hypercube\\ Sampling\\ (LHS)}$', '$\\mathrm{decoupled\\ Latin\\ Hypercube\\ Design\\ (d-LHD)}$']
    for nnum,noise in enumerate(noisearr):
        mean ={}
        sd = {}
        for snum, sample in enumerate(allsamples):
            mean[sample] = []
            sd[sample] = []
            for fname in farr:
                # mean[sample].append(results[sample][fname][noise]['rnoiters'])
                # sd[sample].append(results[sample][fname][noise]['rnoiterssd'])
                mean[sample].append(np.average(dumpr[sample][fname]))
                sd[sample].append(np.std(dumpr[sample][fname]))
        if(len(axarray)>0):
            ax = plt.subplot2grid((totalrow,totalcol), (0,nnum),sharex=axarray[0],sharey=axarray[0])
            axarray.append(ax)
        else:
            ax = plt.subplot2grid((totalrow,totalcol), (0,nnum))
            axarray.append(ax)

        for snum, sample in enumerate(allsamples):
            # print(mean[type])
            if(sample == 'sg'):
                ax.bar(X111+snum*width, np.array(mean[sample]), width,color=color[snum], capsize=3)
            else:
                ax.bar(X111+snum*width, np.array(mean[sample]), width,color=color[snum],align='center',  ecolor=ecolor, capsize=3,label=legendarr[snum])

                ax.vlines(X111+snum*width, np.array(mean[sample]),np.array(mean[sample])+np.array(sd[sample]),label=None)
        ax.set_xticks(X111 + (len(allsamples)-1)*width / 2)

        xlab = []
        for f in farr:
            # print(f)
            xlab.append("\\ref{fn:%s}"%(f))
            # xlab.append("\\ref{%s}"%(f))
            # xlab.append("%s"%(f))
        xlab = ['$A.1.4$','$A.1.7$','$A.1.15$','$A.1.16$','$A.1.17$']
        ax.set_xticklabels(xlab,fontsize = 24)

        plt.tick_params(labelsize=24)
        # ax.set_xlabel("Test functions",fontsize=22)
        # ax.set_ylabel("$\\log_{10}$ [Number of iterations]",fontsize=40)
        ax.set_ylabel("$\\mathrm{Number\\ of\\ iterations}$",fontsize=28)
        ax.label_outer()
    l1 = ax.legend(loc='upper left', ncol=1,fontsize = 24,frameon=False)
    # plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
    plt.tight_layout()
    # plt.savefig("../../log/iterations.png")
    # ffffff.savefig("../../log/iterations.png", bbox_extra_artists=(l1,), bbox_inches='tight')
    ffffff.savefig("../../log/iterations.pdf", bbox_extra_artists=(l1,), bbox_inches='tight')
    plt.clf()
    plt.close('all')

    exit(1)


    # s = "\nsample is %s\n\n"%(sample)
    # if(table_or_latex == "table"):
    #     s+= "WRONG \n\n \t\t\t"
    #     for noise in noisearr:
    #         s+= "%s\t\t\t\t\t\t\t\t\t\t"%(noise)
    #     s+="\n"
    #     for num,noise in enumerate(noisearr):
    #         if(num==0):
    #             s += "\t\tpdof\trdof"
    #         s += "\tPoly App\tRat Apprx\tRat Apprx SIP\t"
    #     s+="\n\n"
    #     for fname in farr:
    #         s += "%s"%(fname)
    #         for num,noise in enumerate(noisearr):
    #             if(num==0):
    #                 s += "\t\t%d\t%d"%(results[fname][noise]["pdof"],results[fname][noise]["rdof"])
    #                 continue
    #             # s += "\t\t%.4f"%(results[fname][noise]["papp"])
    #             # s+="\t"
    #             # s += "\t%.4f"%(results[fname][noise]["rapp"])
    #             # s+="\t"
    #             s += "\t%.2f"%(results[fname][noise]["rappsip"])
    #             s+="\t"
    #             s += "\t%d"%(results[fname][noise]["rnoiters"])
    #             s+="\t"
    #             s += "\t%d"%(results[fname][noise]["rpnnl"])
    #             s+="\t"
    #             s += "\t%d"%(results[fname][noise]["rqnnl"])
    #             s+="\t"
    #             # break
    #         s+="\n"
    # elif(table_or_latex =="latex"):
    #     for fname in farr:
    #         s += "\\ref{fn:%s}"%(fname)
    #         for num,noise in enumerate(noisearr):
    #             if(num==0):
    #                 s+="&%d&%d&%d"%(results[fname][noise]["pdof"],results[fname][noise]["rdof"],results[fname][noise]["rqnnl"])
    #             s += "&%.1f&%.1f"%(results[fname][noise]["papp"],results[fname][noise]["pappsd"])
    #             s += "&%.1f&%.1f"%(results[fname][noise]["rapp"],results[fname][noise]["rappsd"])
    #             s += "&%.1f&%.1f"%(results[fname][noise]["rapprd"],results[fname][noise]["rapprdsd"])
    #             s += "&%.1f&%.1f"%(results[fname][noise]["rappsip"],results[fname][noise]["rappsipsd"])
    #             s += "&%.1f&%.1f"%(results[fname][noise]["rnoiters"],results[fname][noise]["rnoiterssd"])
    #             # s += "&%.1f"%(results[fname][noise]["papp"])
    #             # s += "&%.1f"%(results[fname][noise]["rapp"])
    #             # s += "&%.1f"%(results[fname][noise]["rapprd"])
    #             # s += "&%.1f"%(results[fname][noise]["rappsip"])
    #             # s += "&%.1f"%(results[fname][noise]["rnoiters"])
    #
    #         s+="\\\\\hline\n"
    #
    # print(s)

# python tablecompareall.py f7,f8  0,10-1 2x ../benchmarkdata/f7.txt,../benchmarkdata/f8.txt all,all latex

# python tabletotalcputime.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-1 2x latex

# python tabletotalcputime.py f1,f2 0,10-2,10-6 2x latex
# python tabletotalcputime.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-6,10-2 2x latex

# iterations
# python tabletotalcputime.py  f4,f8,f9,f10,f14,f16,f17,f18,f20,f21,f22 0 2x latex
# python tabletotalcputime.py  f4,f8,f17,f18,f19 0 2x latex

if __name__ == "__main__":
    import os, sys
    if len(sys.argv) != 5:
        print("Usage: {} function noise ts table_or_latex".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise levels")
        sys.exit(1)


    tabletotalcputime(farr, noisearr, sys.argv[3], sys.argv[4])

###########
