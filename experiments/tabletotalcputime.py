
import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
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


def tabletotalcputime(farr,noisearr, ts, table_or_latex):
    print (farr)
    print (noisearr)

    # allsamples = ['mc','lhs','so','sg']
    allsamples = ['lhs','splitlhs','sg']
    # allsamples = ['sg']
    import json
    from apprentice import tools
    results = {}
    for snum, sample in enumerate(allsamples):
        results[sample] = {}
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
                    timerasip.append(np.log10(rappsiptime))
                    iterrasip.append(np.log10(rnoiters))


                    if rappfile:
                        with open(rappfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapptime = datastore['log']['fittime']
                    timera.append(np.log10(rapptime))

                    if rapprdfile:
                        with open(rapprdfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapprdtime = datastore['log']['fittime']
                    timerard.append(np.log10(rapprdtime))


                    if pappfile:
                        with open(pappfile, 'r') as fn:
                            datastore = json.load(fn)
                    papptime = datastore['log']['fittime']
                    pdof = tools.numCoeffsPoly(datastore['dim'],datastore['m'])
                    timepa.append(np.log10(papptime))
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





                    'pdof':pdof,
                    'rdof':rdof,
                    'rpnnl':rpnnl,
                    'rqnnl':rqnnl
                }



        # from IPython import embed
        # embed()

    # print(results)
    baseline = 0.5
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
    legendarr = ['Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR} without degree reduction','Algorithm \\ref{ALG:MVVandQR}' ,'Algorithm \\ref{A:Polyak}']
    color = ['#900C3F','#C70039','#FF5733','#FFC300']
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    for nnum,noise in enumerate(noisearr):
        for snum, sample in enumerate(allsamples):
            mean ={}
            for type in meankeyarr:
                mean[type] = []
            sd = {}
            for type in sdkeyarr:
                sd[type] = []
            for fname in farr:
                for type in meankeyarr:
                    mean[type].append(results[sample][fname][noise][type])
                    # mean[type].append(np.ma.log10(results[sample][fname][noise][type]))
                for type in sdkeyarr:
                    # print(results[sample][fname][noise][type])
                    sd[type].append(results[sample][fname][noise][type])
                    # sd[type].append(np.ma.log10(results[sample][fname][noise][type]))
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
                    ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], capsize=3)
                else:
                    ax.bar(X111+typenum*width, np.array(mean[type])+baseline, width,color=color[typenum], yerr=np.array(sd[sdkey]),align='center',  ecolor=ecolor, capsize=3)
                ax.set_xticks(X111 + (len(meankeyarr)-1)*width / 2)
                xlab = []
                for f in farr:
                    # print(f)
                    # xlab.append("\\ref{fn:%s}"%(f))
                    xlab.append("\\ref{%s}"%(f))
                ax.set_xticklabels(xlab,fontsize = 20)
                ax.set_xlabel("Test functions",fontsize=22)
                ax.set_ylabel("$\\log_{10}$ [CPU time (sec)]",fontsize=22)
                ax.label_outer()

    ffffff.legend((legendarr),
               loc='upper center', ncol=5,bbox_to_anchor=(0.5, 0.99), fontsize = 25,borderaxespad=0.,shadow=False)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
    plt.tight_layout()
    plt.savefig("../../log/cputime.png")
    plt.clf()
    plt.close('all')



    color = ['#FFC300','#FF5733','#900C3F']
    
    ffffff = plt.figure(0,figsize=(45, 20))
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    totalrow = 1
    totalcol = 3
    baseline = 0.1
    axarray = []
    legendarr = ['Latin hypercube sampling', 'Split latin hypercube sampling', 'Sparse grids']
    for nnum,noise in enumerate(noisearr):
        mean ={}
        sd = {}
        for snum, sample in enumerate(allsamples):
            mean[sample] = []
            sd[sample] = []
            for fname in farr:
                mean[sample].append(results[sample][fname][noise]['rnoiters'])
                sd[sample].append(results[sample][fname][noise]['rnoiterssd'])
            if(len(axarray)>0):
                ax = plt.subplot2grid((totalrow,totalcol), (0,nnum),sharex=axarray[0],sharey=axarray[0])
                axarray.append(ax)
            else:
                ax = plt.subplot2grid((totalrow,totalcol), (0,nnum))
                axarray.append(ax)

        for snum, sample in enumerate(allsamples):
            # print(mean[type])
            if(sample == 'sg'):
                ax.bar(X111+snum*width, np.array(mean[sample])+baseline, width,color=color[snum], capsize=3)
            else:
                ax.bar(X111+snum*width, np.array(mean[sample])+baseline, width,color=color[snum], yerr=np.array(sd[sample]),align='center',  ecolor=ecolor, capsize=3)
            ax.set_xticks(X111 + (len(sample)-1)*width / 2)
            xlab = []
            for f in farr:
                # print(f)
                # xlab.append("\\ref{fn:%s}"%(f))
                xlab.append("\\ref{%s}"%(f))
            ax.set_xticklabels(xlab,fontsize = 20)
            ax.set_xlabel("Test functions",fontsize=22)
            ax.set_ylabel("$\\log_{10}$ [Number of iterations]",fontsize=22)
            ax.label_outer()
    ffffff.legend((legendarr),
               loc='upper center', ncol=5,bbox_to_anchor=(0.5, 0.99), fontsize = 25,borderaxespad=0.,shadow=False)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
    plt.tight_layout()
    plt.savefig("../../log/iterations.png")
    plt.clf()
    plt.close('all')

    exit(1)


    s = "\nsample is %s\n\n"%(sample)
    if(table_or_latex == "table"):
        s+= "WRONG \n\n \t\t\t"
        for noise in noisearr:
            s+= "%s\t\t\t\t\t\t\t\t\t\t"%(noise)
        s+="\n"
        for num,noise in enumerate(noisearr):
            if(num==0):
                s += "\t\tpdof\trdof"
            s += "\tPoly App\tRat Apprx\tRat Apprx SIP\t"
        s+="\n\n"
        for fname in farr:
            s += "%s"%(fname)
            for num,noise in enumerate(noisearr):
                if(num==0):
                    s += "\t\t%d\t%d"%(results[fname][noise]["pdof"],results[fname][noise]["rdof"])
                    continue
                # s += "\t\t%.4f"%(results[fname][noise]["papp"])
                # s+="\t"
                # s += "\t%.4f"%(results[fname][noise]["rapp"])
                # s+="\t"
                s += "\t%.2f"%(results[fname][noise]["rappsip"])
                s+="\t"
                s += "\t%d"%(results[fname][noise]["rnoiters"])
                s+="\t"
                s += "\t%d"%(results[fname][noise]["rpnnl"])
                s+="\t"
                s += "\t%d"%(results[fname][noise]["rqnnl"])
                s+="\t"
                # break
            s+="\n"
    elif(table_or_latex =="latex"):
        for fname in farr:
            s += "\\ref{fn:%s}"%(fname)
            for num,noise in enumerate(noisearr):
                if(num==0):
                    s+="&%d&%d&%d"%(results[fname][noise]["pdof"],results[fname][noise]["rdof"],results[fname][noise]["rqnnl"])
                s += "&%.1f&%.1f"%(results[fname][noise]["papp"],results[fname][noise]["pappsd"])
                s += "&%.1f&%.1f"%(results[fname][noise]["rapp"],results[fname][noise]["rappsd"])
                s += "&%.1f&%.1f"%(results[fname][noise]["rapprd"],results[fname][noise]["rapprdsd"])
                s += "&%.1f&%.1f"%(results[fname][noise]["rappsip"],results[fname][noise]["rappsipsd"])
                s += "&%.1f&%.1f"%(results[fname][noise]["rnoiters"],results[fname][noise]["rnoiterssd"])
                # s += "&%.1f"%(results[fname][noise]["papp"])
                # s += "&%.1f"%(results[fname][noise]["rapp"])
                # s += "&%.1f"%(results[fname][noise]["rapprd"])
                # s += "&%.1f"%(results[fname][noise]["rappsip"])
                # s += "&%.1f"%(results[fname][noise]["rnoiters"])

            s+="\\\\\hline\n"

    print(s)

# python tablecompareall.py f7,f8  0,10-1 2x ../benchmarkdata/f7.txt,../benchmarkdata/f8.txt all,all latex

# python tabletotalcputime.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-1 2x latex

# python tabletotalcputime.py f1,f2 0,10-2,10-6 2x latex
# python tabletotalcputime.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-6,10-2 2x latex

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
