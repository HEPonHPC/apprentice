# Algorithm from www.1728.org/cubic2.htm

import math
import numpy as np

# Polynomial Structure  ax^3 + bx^2 + cx + d = 0

def solve(a, b, c, d):

    if (a == 0 and b == 0):
        return np.array([(-d * 1.0) / c])

    elif (a == 0):

        D = c * c - 4.0 * b * d
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)

        return np.array([x1, x2])

    f = findF(a, b, c)
    g = findG(a, b, c, d)
    h = findH(g, f)

    if f == 0 and g == 0 and h == 0:
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x]),"threeequal"

    elif h <= 0:

        i = math.sqrt(((g ** 2.0) / 4.0) - h)
        j = i ** (1 / 3.0)
        k = math.acos(-(g / (2 * i)))
        L = j * -1
        M = math.cos(k / 3.0)
        N = math.sqrt(3) * math.sin(k / 3.0)
        P = (b / (3.0 * a)) * -1

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3]),"threereal"

    elif h > 0:
        R = -(g / 2.0) + math.sqrt(h)
        if R >= 0:
            S = R ** (1 / 3.0)
        else:
            S = (-R) ** (1 / 3.0) * -1
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))
        else:
            U = ((-T) ** (1 / 3.0)) * -1

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        return np.array([x1, x2, x3]),"onereal"



def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

def findsincroots():
    import os, sys
    def appevaldenom(x,y,z,app,toscale,printnumer):
        if(toscale ==1):
            X = app._scaler.scale(np.array([x,y,z]))
        else:
            X = np.array([x,y,z])
        if(printnumer==1):
            print("numer=%f"%(app.numer(X)))

        return app.denom(X)

    def appevalnumer(x,y,z,app,toscale,printnumer):
        if(toscale ==1):
            X = app._scaler.scale(np.array([x,y,z]))
        else:
            X = np.array([x,y,z])
        n = app.numer(X)
        if(printnumer==1):
            print("numer=%f"%(n))

        return n

    def appgetonevardenom(x,y,app):
        import apprentice
        struct_q = apprentice.monomialStructure(datastore['dim'], datastore['n'])
        qcoeff = app.qcoeff
        a=0
        b=0
        c=0
        d=0
        for num,row in enumerate(struct_q):
            val = qcoeff[num]*(x**row[0] * y**row[1])
            if(row[2]==0):
                d+=val
            elif(row[2]==1):
                c+=val
            elif(row[2]==2):
                b+=val
            elif(row[2]==3):
                a+=val

        # z = -0.2345
        # fthis = a*z**3 + b*z**2 +c*z +d
        # fnorm = appevaldenom(x,y,z,app,0,0)
        # print(fthis,fnorm)
        return a,b,c,d




    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}
    m=2
    n=3
    ts="2x"

    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    dim = 3
    numlb = 0
    numub = 1
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
    l = "_p%d_q%d_ts%s.json"%(m,n, ts)
    jsonfile = folder+'/'+fndesc+"/out/"+fndesc+l
    if not os.path.exists(jsonfile):
        print("%s not found"%(jsonfile))
        exit(1)
    if not os.path.exists(folder+'/'+fndesc+"/plots"):
        os.mkdir(folder+'/'+fndesc+'/plots')
    import json
    if jsonfile:
        with open(jsonfile, 'r') as fn:
            datastore = json.load(fn)
    for iter in range(len(datastore['iterationinfo'])):
        # print("Doing iter = %d"%(iter+1))

        if jsonfile:
            with open(jsonfile, 'r') as fn:
                datastore = json.load(fn)
        datastore['pcoeff'] = datastore['iterationinfo'][iter]['pcoeff']
        datastore['qcoeff'] = datastore['iterationinfo'][iter]['qcoeff']
        from apprentice import RationalApproximationSIP
        rappsip = RationalApproximationSIP(datastore)
        robarg = datastore['iterationinfo'][iter]['robOptInfo']['robustArg']
        # print(rappsip._scaler.unscale(np.array(robarg)))
        print(robarg)
        continue

        X = np.linspace(-1, 1, num=1000)
        Y = np.linspace(-1, 1, num=1000)
        X0 = np.array([])
        Y0 = np.array([])
        Z0 = np.array([])
        N = np.array([])
        for x in X:
            for y in Y:
                a,b,c,d = appgetonevardenom(x,y,rappsip)
                roots,res = solve(a,b,c,d)
                if(res == "threeequal" or res == "onereal"):
                    r = np.real(roots[0])
                    if(r>=-1 and r <=1):
                        xu,yu,ru = rappsip._scaler.unscale([x,y,r])
                        qnormu = appevaldenom(xu,yu,ru,rappsip,1,0)
                        if(qnormu == 0):
                            X0 = np.append(X0,xu)
                            Y0 = np.append(Y0,yu)
                            Z0 = np.append(Z0,ru)
                            N = np.append(N,appevalnumer(xu,yu,ru,rappsip,1,0))
                elif(res == 'threereal'):
                    for r in roots:
                        r = np.real(r)
                        if(r>=-1 and r <=1):
                            xu,yu,ru = rappsip._scaler.unscale([x,y,r])
                            qnormu = appevaldenom(xu,yu,ru,rappsip,1,0)
                            if(qnormu == 0):
                                X0 = np.append(X0,xu)
                                Y0 = np.append(Y0,yu)
                                Z0 = np.append(Z0,ru)
                                N = np.append(N,appevalnumer(xu,yu,ru,rappsip,1,0))

        outcsv = "%s/%s/plots/Croots_iter%d.csv"%(folder, fndesc, iter+1)
        np.savetxt(outcsv,np.stack((X0,Y0,Z0,N),axis=1), delimiter=",")
        # print("CSV written to %s"%(outcsv))


if __name__ == "__main__":
    x = solve(1, -3, 4, -4)
    y = solve(1, 0, -21, -20)
    print(x)
    print(y)

    findsincroots()
