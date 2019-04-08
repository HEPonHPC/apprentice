
import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
import os


def tabletotalcputime(farr,noisearr, ts, table_or_latex):
    print (farr)
    print (noisearr)

    results = {}

    # import glob
    import json
    # import re
    for num,fname in enumerate(farr):
        results[fname] = {}
        for noise in noisearr:
            results[fname][noise] = {}
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)

            optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"

            if not os.path.exists(optjsonfile):
                print("optjsonfile: " + optjsonfile+ " not found")
                exit(1)

            if optjsonfile:
                with open(optjsonfile, 'r') as fn:
                    optjsondatastore = json.load(fn)

            optm = optjsondatastore['optdeg']['m']
            optn = optjsondatastore['optdeg']['n']
            break
        for noise in noisearr:
            results[fname][noise] = {}
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)
            rappsipfile = "%s/out/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
            rappfile = "%s/outra/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
            pafile = "%s/outpa/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)

            if not os.path.exists(rappsipfile):
                print("rappsipfile %s not found"%(rappsipfile))
                exit(1)

            if not os.path.exists(rappfile):
                print("rappfile %s not found"%(rappfile))
                exit(1)

            if not os.path.exists(pafile):
                print("pappfile %s not found"%(pafile))
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


            if rappfile:
                with open(rappfile, 'r') as fn:
                    datastore = json.load(fn)
            rapptime = datastore['log']['fittime']

            if pafile:
                with open(pafile, 'r') as fn:
                    datastore = json.load(fn)
            papptime = datastore['log']['fittime']
            pdof = int(datastore['trainingsize']/2)

            results[fname][noise] = {"rapp":rapptime, "rappsip":rappsiptime,
            "papp":papptime,'pdof':pdof,'rdof':rdof,'rnoiters':rnoiters,
            'rpnnl':rpnnl,'rqnnl':rqnnl}


    # from IPython import embed
    # embed()


    s = ""
    if(table_or_latex == "table"):
        s+= "\t\t\t"
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
                s += "&%.3f"%(results[fname][noise]["papp"])
                s += "&%.3f"%(results[fname][noise]["rapp"])
                s += "&%.3f"%(results[fname][noise]["rappsip"])
                s += "&%d"%(results[fname][noise]["rnoiters"])
            s+="\\\\\hline\n"

    print(s)

# python tablecompareall.py f7,f8  0,10-1 2x ../benchmarkdata/f7.txt,../benchmarkdata/f8.txt all,all latex
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
