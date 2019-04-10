
import apprentice
import numpy as np
import os, sys
import json

def tablesinc(m,n,ts,table_or_latex):
    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}
    lblatex = {0:"$10^{-6}$",1:"$10^{-3}$"}
    ublatex = {0:"$2\\pi$",1:"$4\\pi$"}


    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        print("folder %s not found")

    if not os.path.exists(folder+"/benchmarkdata"):
        os.mkdir(folder+'/benchmarkdata')

    data = {}
    for dim in range(2,8):
        data[dim] = {}
        for numlb,lb in enumerate(larr):
            for numub,ub in enumerate(uarr):
                key = lbdesc[numlb]+ubdesc[numub]
                data[dim][key] = {}
                fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])

                file = folder+"/"+fndesc+'/out/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".json"
                if not os.path.exists(file):
                    print("%s not found"%(file))

                if file:
                    with open(file, 'r') as fn:
                        datastore = json.load(fn)
                rappsiptime = datastore['log']['fittime']
                rdof = int(datastore['M'] + datastore['N'])
                rnoiters = len(datastore['iterationinfo'])
                rpnnl = datastore['M'] - (dim+1)
                rqnnl = datastore['N'] - (dim+1)
                data[dim][key]['rappsiptime'] = rappsiptime
                data[dim][key]['rdof'] = rdof
                data[dim][key]['rnoiters'] = rnoiters
                data[dim][key]['rpnnl'] = rappsiptime
                data[dim][key]['rqnnl'] = rqnnl
    # print(data)

    s =""
    if(table_or_latex == "table"):
        print("TBD")
    elif(table_or_latex =="latex"):
        for dim in range(2,8):
            for numlb,lb in enumerate(larr):
                for numub,ub in enumerate(uarr):

                    key = lbdesc[numlb]+ubdesc[numub]
                    s += "%d&%d&%d&%s&%s&%.3f&%d"%(dim,data[dim][key]['rdof'],data[dim][key]['rqnnl'],
                                lblatex[numlb],ublatex[numub],data[dim][key]['rappsiptime'],data[dim][key]['rnoiters'])
                    s+="\\\\\hline\n"
    # print(s)

    import matplotlib.pyplot as plt
    X = range(2,8)
    rangearr = []
    labelarr = []
    for numlb,lb in enumerate(larr):
        for numub,ub in enumerate(uarr):
            rangearr.append(lbdesc[numlb]+ubdesc[numub])
            labelarr.append(lblatex[numlb]+ " - "+ ublatex[numub])
    for r in rangearr:
        Y = []
        for x in X:
            Y.append(data[x][r]['rnoiters'])
        plt.plot(X,np.log10(Y), linewidth=1)
    plt.legend(labelarr,loc='upper left')
    # plt.show()
    plt.savefig("/Users/mkrishnamoorthy/Desktop/sinc.pdf")








if __name__ == "__main__":

    if len(sys.argv)!=5:
        print("Usage: {} m n ts table_or_latex".format(sys.argv[0]))
        sys.exit(1)


    tablesinc(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])
