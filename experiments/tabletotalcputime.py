
import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
import os



def tabletotalcputime(farr,noisearr, ts, table_or_latex):
    print (farr)
    print (noisearr)



    # allsamples = ['mc','lhs','sc','sg']
    allsamples = ['sg']
    # import glob
    import json
    # import re
    for snum, sample in enumerate(allsamples):
        results = {}
        for num,fname in enumerate(farr):
            results[fname] = {}
            # for noise in noisearr:
            #     results[fname][noise] = {}
            #     noisestr = ""
            #     if(noise!="0"):
            #         noisestr = "_noisepct"+noise
            #     folder = "%s%s_%s"%(fname,noisestr,ts)
            #
            #     optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
            #
            #     if not os.path.exists(optjsonfile):
            #         print("optjsonfile: " + optjsonfile+ " not found")
            #         exit(1)
            #
            #     if optjsonfile:
            #         with open(optjsonfile, 'r') as fn:
            #             optjsondatastore = json.load(fn)
            #
            #     optm = optjsondatastore['optdeg']['m']
            #     optn = optjsondatastore['optdeg']['n']
            #     break
            m = 5
            n = 5

            for noise in noisearr:
                results[fname][noise] = {}
                noisestr = ""
                if(noise!="0"):
                    noisestr = "_noisepct"+noise

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
                        exit(1)

                    if not os.path.exists(rappfile):
                        print("rappfile %s not found"%(rappfile))
                        exit(1)

                    if not os.path.exists(rapprdfile):
                        print("rapprdfile %s not found"%(rapprdfile))
                        exit(1)

                    if not os.path.exists(pappfile):
                        print("pappfile %s not found"%(pappfile))
                        exit(1)

                    if rappsipfile:
                        with open(rappsipfile, 'r') as fn:
                            datastore = json.load(fn)
                    rappsiptime = datastore['log']['fittime']
                    rdof = int(datastore['M'] + datastore['N'])
                    rnoiters = len(datastore['iterationinfo'])
                    dim = datastore['dim']
                    rpnnl = datastore['M'] - (dim+1)
                    rqnnl = datastore['N'] - (dim+1)
                    timerasip.append(rappsiptime)
                    iterrasip.append(rnoiters)


                    if rappfile:
                        with open(rappfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapptime = datastore['log']['fittime']
                    timera.append(rapptime)

                    if rapprdfile:
                        with open(rapprdfile, 'r') as fn:
                            datastore = json.load(fn)
                    rapprdtime = datastore['log']['fittime']
                    timerard.append(rapprdtime)


                    if pappfile:
                        with open(pappfile, 'r') as fn:
                            datastore = json.load(fn)
                    papptime = datastore['log']['fittime']
                    pdof = tools.numCoeffsPoly(datastore['dim'],datastore['m'])
                    timepa.append(papptime)
                    if(sample == "sg"):
                        break

                results[fname][noise] = {
                    "rapprd":np.average(timerard),
                    "rapprdsd":np.std(timerard),
                    "rapp":np.average(timera),
                    "rappsip":np.average(timerasip),
                    'rnoiters':np.average(iterrasip),
                    "papp":np.average(timepa),
                    "rappsd":np.std(timera),
                    "rappsipsd":np.std(timerasip),
                    'rnoiterssd':np.std(iterrasip),
                    "pappsd":np.std(timepa),
                    'pdof':pdof,
                    'rdof':rdof,
                    'rpnnl':rpnnl,
                    'rqnnl':rqnnl}



        # from IPython import embed
        # embed()


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
                    # s += "&%.1f&%.1f"%(results[fname][noise]["papp"],results[fname][noise]["pappsd"])
                    # s += "&%.1f&%.1f"%(results[fname][noise]["rapp"],results[fname][noise]["rappsd"])
                    # s += "&%.1f&%.1f"%(results[fname][noise]["rapprd"],results[fname][noise]["rapprdsd"])
                    # s += "&%.1f&%.1f"%(results[fname][noise]["rappsip"],results[fname][noise]["rappsipsd"])
                    # s += "&%.1f&%.1f"%(results[fname][noise]["rnoiters"],results[fname][noise]["rnoiterssd"])
                    s += "&%.1f"%(results[fname][noise]["papp"])
                    s += "&%.1f"%(results[fname][noise]["rapp"])
                    s += "&%.1f"%(results[fname][noise]["rapprd"])
                    s += "&%.1f"%(results[fname][noise]["rappsip"])
                    s += "&%.1f"%(results[fname][noise]["rnoiters"])

                s+="\\\\\hline\n"

        print(s)

# python tablecompareall.py f7,f8  0,10-1 2x ../benchmarkdata/f7.txt,../benchmarkdata/f8.txt all,all latex
# python tabletotalcputime.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-1 2x latex
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
