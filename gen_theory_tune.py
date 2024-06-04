#!/usr/bin/env python3

from apprentice.function import Function
from apprentice.polyset import PolySet
from apprentice.polynomialapproximation import PolynomialApproximation
from apprentice.rationalapproximation import RationalApproximation
from apprentice.generatortuning import GeneratorTuning
from apprentice.scipyminimizer import ScipyMinimizer
from apprentice.weights import read_pointmatchers

from apprentice.util import Util
import apprentice.io as IO
import apprentice.tools as TOOLS
import numpy as np
import time

class TheoryErrorTuning(GeneratorTuning):
    """
    Generator Tuning with Theory Errors
    """

    def __init__(self, dim, fnspace, data, errors, s_val, e_val, weights, binids, bindn, binup, **kwargs):
        """
        :param dim: parameter dimension
        :type dim: int
        :param fnspace: function space object
        :type fnspace: apprentice.space.Space
        :param data: data values
        :type data: np.array
        :param s_val: surrogate polyset or list of SurrogateModel
        :type s_val: apprentice.polyset.PolySet or list
        :param errors: data errors
        :type errors: np.array
        :param weights: weights
        :type weights: np.array
        :param e_val: surrogate polyset or list of SurrogateModel
        :type e_val: apprentice.polyset.PolySet or list
        :param binids: list of binids
        :type binids: list
        :param bindn: bin down
        :type bindn: list
        :param binup: bin up
        :type binup: list

        """

        super(TheoryErrorTuning, self).__init__(dim, fnspace, data, errors, s_val, e_val, weights, binids, bindn, binup, **kwargs)
        self.binids_  = binids
        self.hnames_  = np.array([b.split("#")[0]  for b in self.binids_])
        self.bindn_   = bindn
        self.binup_   = binup
        self.eps_         = kwargs['RPARAM']
        self.theoryerr_   = kwargs['THERR']

    # Adapted from notebook shared by Enzo
    # Minimize the likelihood with respect to nuisance parameters analytically, in real application this step has often to be done numerically
    # This is the _very_ slow version
    def theta_profile(self, mu, y, u, sigma_u, sigma_y_sq, r):
        import math
        root3 = math.sqrt(3)
        oneover3 = 1.0/3.0
        v = sigma_u**2
        #a = 1.
        # u = 0
        b = - y + mu
        tworsq = 2. * r * r
        c = (v+(1.+tworsq)*sigma_y_sq)/tworsq
        d = b * v / tworsq 
        f = c  - b * b * oneover3             # Helper Temporary Variable
        g = 2.0 * b * b * b / 27. - b * c * oneover3 + d     # Helper Temporary Variable                    
        h = g * g / 4.0 + f * f * f / 27.0               # Helper Temporary Variable

        # theta0 = np.array([theta_profile(expected_values[i], Xsec[i], u[i], sigma_u[i], sigma_Xsec[i], eps[i]) for i in range(len(Xsec))])
        #    if f == 0 and g == 0:
        #        if d  >= 0:
        #            x = - np.cbrt(d)
        #        else:
        #            x = np.cbrt(-d)
        #        theta = x

        if h > 0.:                                       # One Real Root and two Complex Roots 
            R = - 0.5 * g  + math.sqrt(h)                 # Helper Temporary Variable
            if R >= 0:
                S = R ** oneover3
            else:
                S = - (-R) ** oneover3
            T = - 0.5 * g  - math.sqrt(h)
            if T >= 0:
                U = T ** oneover3
            else:
                U = - (-T) ** oneover3

            x = S + U - b * oneover3
            theta = x

        else:
            i = math.sqrt(0.25* g * g - h)         # Helper Temporary Variable
            #j = np.cbrt(i)
            k = math.acos(-0.5 * g / i)                # Helper Temporary Variable
            #L = - np.cbrt(i)     # Helper Temporary Variable
            L = - i ** oneover3
            M = math.cos(k * oneover3)                         # Helper Temporary Variable
            N = root3 * math.sin(k * oneover3)          # Helper Temporary Variable
            P = - b * oneover3                         # Helper Temporary Variable

            x1 = - 2. * L * math.cos(k * oneover3) - b * oneover3
            x2 = L * (M + N) + P
            x3 = L * (M - N) + P

            x = np.array([x1, x2, x3])
            lik = - 0.5 *(y - mu - x)**2/sigma_y_sq - 0.5*(1+1./tworsq*np.log(1.+tworsq*(u - x)**2/sigma_u**2))
            index = np.argmin(lik)
            theta = x[index]
        return theta

    def theta_profile_vec(self, mu, y, u, sigma_u, sigma_y_sq, r):
        root3 = np.sqrt(3)
        oneover3 = 1.0/3.0
        v = sigma_u**2
        #a = 1.
        # u = 0
        b = - y + mu
        tworsq = 2. * r * r
        c = (v+(1.+tworsq)*sigma_y_sq)/tworsq
        d = b * v / tworsq 
        f = c  - b * b * oneover3             # Helper Temporary Variable
        g = 2.0 * b * b * b / 27. - b * c * oneover3 + d     # Helper Temporary Variable                    
        h = 0.25 * g * g + f * f * f / 27.0               # Helper Temporary Variable

        g_prime = np.where( h < 0, 0, g)
        h_prime = np.where( h < 0, 0, h)
        #Rmatch = np.ma.masked_where( h < 0, -0.5 * g_prime + np.sqrt( h_prime ) )
        #Tmatch = np.ma.masked_where( h < 0, -0.5 * g_prime - np.sqrt( h_prime ) )
        Rmatch = -0.5 * g_prime + np.sqrt( h_prime ) 
        Tmatch = -0.5 * g_prime - np.sqrt( h_prime ) 
        """
        RplusRt  = np.ma.masked_where( Rmatch < 0, Rmatch ** oneover3 )
        RminusRt  = np.ma.masked_where( Rmatch <= 0, - ( -Rmatch ) ** oneover3 )
        TminusRt = np.ma.masked_where( Tmatch > 0, - ( -Tmatch ) ** oneover3 )
        TplusRt = np.ma.masked_where( Tmatch <= 0, Tmatch ** oneover3 )

        theta0  += np.ma.filled(RplusRt,0) + np.ma.filled(TminusRt,0) + np.ma.filled(RminusRt,0) + np.ma.filled(TplusRt,0)
        """
        #theta0 += np.ma.filled( np.sign(Rmatch)*np.absolute(Rmatch) ** oneover3 + np.sign(Tmatch)*np.absolute(Tmatch) ** oneover3, 0 )
        #theta0  = np.ma.masked_where( h < 0, -b * oneover3 )
        #theta0 += np.sign(Rmatch)*np.absolute(Rmatch) ** oneover3 + np.sign(Tmatch)*np.absolute(Tmatch) ** oneover3
        theta0 = np.ma.masked_where( h < 0 , np.sign(Rmatch)*np.absolute(Rmatch) ** oneover3 + np.sign(Tmatch)*np.absolute(Tmatch) ** oneover3
                                     - b * oneover3 )

        #theta    = np.ma.filled(theta0, 0)
        g_prime = np.where( h >= 0, 0, g)
        h_prime = np.where( h >= 0, 0, h)
        iFactor = np.ma.masked_where( h >= 0, np.sqrt(0.25 * g_prime * g_prime - h_prime))
        kFactor = np.ma.masked_where( h >= 0, np.arccos( -0.5 * g_prime / iFactor) * oneover3 )
        lFactor = np.ma.masked_where( h >= 0, - iFactor ** oneover3 )
        mFactor = np.ma.masked_where( h >= 0, np.cos( kFactor ) )
        nFactor = np.ma.masked_where( h >= 0, root3 * np.sin( kFactor) )
        bMasked = np.ma.masked_where( h >= 0, - b * oneover3 )

        x = np.empty((3,) + h.shape)
        x[0] = -2.0*lFactor*mFactor + bMasked
        x[1] = lFactor*(mFactor+nFactor) + bMasked
        x[2] = lFactor*(mFactor-nFactor) + bMasked
        lik = - (y - mu - x)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq * x**2/v)
        idx = np.argmin(lik, axis=0, keepdims=True) # indices of the minimum lik-s
        theta_case2 = np.take_along_axis(x, indices=idx, axis=0).squeeze(axis=0)

        theta = np.where( h >= 0, theta0, theta_case2)

        """
        x1 = np.ma.masked_where( h >= 0, - 2.0 * lFactor * mFactor ) + bMasked
        x2 = np.ma.masked_where( h >= 0, lFactor * ( mFactor + nFactor ) ) + bMasked
        x3 = np.ma.masked_where( h >= 0, lFactor * ( mFactor - nFactor ) ) + bMasked

        # remove universal factors
        lik1 = - (y - mu - x1)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x1**2/sigma_u**2)
        lik2 = - (y - mu - x2)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x2**2/sigma_u**2)
        lik3 = - (y - mu - x3)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x3**2/sigma_u**2)

        xstar1 = np.ma.masked_where( np.logical_or(lik1 > lik2, lik1 > lik3), x1 )
        xstar2 = np.ma.masked_where( np.logical_or(lik2 > lik1, lik2 > lik3), x2 )
        xstar3 = np.ma.masked_where( np.logical_or(lik3 > lik2, lik3 > lik1), x3 )

        theta += np.ma.filled(xstar1,0) + np.ma.filled(xstar2,0) + np.ma.filled(xstar3,0)
        """
        return theta

    def theta_profile_vec_simple(self, mu, y, u, sigma_u, sigma_y_sq, r):
        root3 = np.sqrt(3)
        oneover3 = 1.0/3.0
        v = sigma_u**2
        #a = 1.
        #u = 0
        b = - y + mu
        tworsq = 2. * r * r
        c = (v+(1.+tworsq)*sigma_y_sq)/tworsq
        d = b * v / tworsq 
        f = c  - b * b * oneover3             # Helper Temporary Variable
        g = 2.0 * b * b * b / 27. - b * c * oneover3 + d     # Helper Temporary Variable                    
        h = 0.25 * g * g + f * f * f / 27.0               # Helper Temporary Variable

        theta = np.zeros_like( h )
        g_prime = np.where( h < 0, 0, g)
        h_prime = np.where( h < 0, 0, h)
        b_prime = np.where( h < 0, 0, -b * oneover3 )

        Rmatch = -0.5 * g_prime + np.sqrt( h_prime )
        Tmatch = -0.5 * g_prime - np.sqrt( h_prime )

        RpmRt = np.sign(Rmatch)*np.absolute(Rmatch) ** oneover3
        TpmRt = np.sign(Tmatch)*np.absolute(Tmatch) ** oneover3

        theta += b_prime + RpmRt + TpmRt

        f_prime = np.where( h >= 0, -1, f)
        g_prime = np.where( h >= 0,  1, g)
        iFactor = np.where( h >= 0,  1, np.sqrt( - (f_prime*oneover3)**3 ) )
        kFactor = np.where( h >= 0, 0, np.arccos( -0.5 * g_prime / iFactor ) * oneover3 )
        lFactor = np.where( h >= 0, 0, - iFactor ** oneover3 )
        mFactor = np.where( h >= 0, 0, np.cos( kFactor ) )
        nFactor = np.where( h >= 0, 0, root3 * np.sin( kFactor) )

        #minus_b_over_3 = np.where( h >= 0, 0, - b * oneover3 )
        x = np.empty((3,) + h.shape)
        x[0] = -2.0*lFactor*mFactor + b_prime
        x[1] = lFactor*(mFactor+nFactor) + b_prime
        x[2] = lFactor*(mFactor-nFactor) + b_prime
        lik = - (y - mu - x)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq * x**2/sigma_u**2)
        idx = np.argmin(lik, axis=0, keepdims=True) # indices of the minimum lik-s
        theta_case2 = np.take_along_axis(x, indices=idx, axis=0).squeeze(axis=0)

        theta = np.where( h >= 0, theta, theta_case2)

        """
        bMasked = np.where( h >= 0, 0, - b * oneover3 )

        x1 = np.ma.masked_where( h >= 0, - 2.0 * lFactor * mFactor + bMasked )
        x2 = np.ma.masked_where( h >= 0, lFactor * ( mFactor + nFactor ) + bMasked )
        x3 = np.ma.masked_where( h >= 0, lFactor * ( mFactor - nFactor ) + bMasked )

        # remove universal factors
        lik1 = - (y - mu - x1)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x1**2/sigma_u**2)
        lik2 = - (y - mu - x2)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x2**2/sigma_u**2)
        lik3 = - (y - mu - x3)**2/sigma_y_sq - 1./tworsq*np.log(1. + tworsq* x3**2/sigma_u**2)

        xstar1 = np.ma.masked_where( np.logical_or(lik1 > lik2, lik1 > lik3), x1 )
        xstar2 = np.ma.masked_where( np.logical_or(lik2 > lik1, lik2 > lik3), x2 )
        xstar3 = np.ma.masked_where( np.logical_or(lik3 > lik2, lik3 > lik1), x3 )
        theta += np.ma.filled(xstar1,0) + np.ma.filled(xstar2,0) + np.ma.filled(xstar3,0)
        """


        return theta
 
    def objective(self, x, unbiased=False):
        """

        Compute least squares objective function value at new data point x

        :param x: a new x point, an array of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
         :return: least squares objective function value at new data point x
         :rtype: float

        """
        denom = np.array(self.err2_)
        if self.evals is not None:  denom += np.array(self.evals(x))**2

        eps   = self.eps_ * np.ones_like( self.data_)

        sigma_u = self.theoryerr_ * np.array( self.vals(x) )
        u     = np.zeros_like( eps )

        #theta = self.theta_profile( np.array( self.vals(x) ), self.data_, u, sigma_u, denom, eps)
        #theta = np.array([self.theta_profile(self.vals(x)[i], self.data_[i], u[i], sigma_u[i], denom[i], eps[i]) for i in range(len(self.data_))])
        theta = self.theta_profile_vec( np.array( self.vals(x) ), self.data_, u, sigma_u, denom, eps)

        #theta = self.theta_profile_vec_simple( np.array( self.vals(x) ), self.data_, u, sigma_u, denom, eps)

        y_term = ( self.data_ - np.array(self.vals(x)) -theta)**2 / denom 
        u_term = (1. + 1./(2*eps**2)) * np.log(1.+2.*eps**2*(u - theta)**2/sigma_u**2)

        return np.sum(y_term + u_term) if unbiased else np.sum( np.sqrt(self.prf2_) *( y_term + u_term) )





def lineScan(TO, x0, dim, npoints=100, bounds=None):
    if bounds is None:
        xmin, xmax = TO.bounds_[TO.free_indices_][dim]
    else:
        xmin, xmax = bounds

    xcoords = list(np.linspace(xmin, xmax, npoints))
    xcoords.append(x0[dim])
    xcoords.sort()

    X = np.tile(x0, (len(xcoords),1))
    for num, x in enumerate(X):
        x[dim] = xcoords[num]
    return X

def mkPlotsMinimum(TO, x0, y0=None, prefix=""):
    import pylab
    pnames = np.array(TO.fnspace_.pnames)[TO.free_indices_]

    XX = [lineScan(TO,x0,i) for i in range(len(x0))]
    YY = []
    for i, X in enumerate(XX):
        Y   =[TO.objective(x) for x in X]
        YY.append(Y)
        Y   =[TO.objective(x, unbiased=True) for x in X]
        YY.append(Y)
    ymax=np.max(np.array(YY))
    ymin=np.min(np.array(YY))

    for i in range(len(x0)):
        pylab.clf()
        X=lineScan(TO,x0,i)
        Y   =[TO.objective(x) for x in X]
        Yunb=[TO.objective(x, unbiased=True) for x in X]
        pylab.axvline(x0[i], label="x0=%.5f"%x0[i], color="k")
        y0=TO.objective(x0, unbiased=True)
        # pylab.axhline(y0, label="unbiased y0=%.2f"%y0, color="k")
        pylab.plot(X[:,i], Y, label="objective")
        # pylab.plot(X[:,i], Yunb, linestyle="dashed", label="unbiased")
        #TODO just normalized ratio of bias vs unbiased
        pylab.ylabel("objective")
        pylab.xlabel(pnames[i])
        pylab.ylim((0.9*ymin, 1.1*ymax))
        # if abs(ymin-ymax)>1000:
            # pylab.yscale("log")
        pylab.legend()
        pylab.tight_layout()
        pylab.savefig(prefix+"valley_{}.pdf".format(i))

def mkPlotsCorrelation(TO, x0, prefix=""):
    H=TO.hessian(x0)
    COV = np.linalg.inv(H)
    std_ = np.sqrt(np.diag(COV))
    COR = COV / np.outer(std_, std_)

    nd = len(x0)
    mask =  np.tri(COR.shape[0], k=0)
    A = np.ma.array(COR, mask=mask)
    import pylab
    pylab.clf()
    bb = pylab.imshow(A, vmin=-1, vmax=1, cmap="RdBu")
    locs, labels = pylab.yticks()
    pylab.yticks([i for i in range(nd)], TO.fnspace_.pnames, rotation=00)
    locs, labels = pylab.xticks()
    pylab.xticks([i for i in range(nd)], TO.fnspace_.pnames, rotation=90)
    cbar = pylab.colorbar(bb, extend='both')

    pylab.tight_layout()
    pylab.savefig(prefix+"corr.pdf")

def printParams(TO, x):
    slen = max((max([len(p) for p in TO.fnspace_.pnames]), 6))
    from apprentice.appset import dot_aligned
    x_aligned = dot_aligned(x)
    plen = max((max([len(p) for p in x_aligned]), 6))

    b_dn = dot_aligned(TO.bounds_[:,0])
    b_up = dot_aligned(TO.bounds_[:,1])
    dnlen = max((max([len(p) for p in b_dn]), 5))
    uplen = max((max([len(p) for p in b_up]), 6))

    islowbound = x==TO.bounds_[:,0]
    isupbound  = x==TO.bounds_[:,1]
    isbound = islowbound + isupbound

    isbelow = x < TO.fnspace_.box[:,0]
    isabove = x > TO.fnspace_.box[:,1]
    isoutside = isbelow + isabove

    #isfixed = [i in self._fixIdx[0] for i in range(self.dim)]
    isfixed = [i in TO.fixed_indices_ for i in range(TO.fnspace_.dim_)]

    s= ""
    s+= ("#\n#{:<{slen}}\t{:<{plen}} #    COMMENT    [ {:<{dnlen}}  ...  {:<{uplen}} ]\n#\n".format(" PNAME", " PVALUE", " PLOW", " PHIGH", slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
    for pn, val, bdn, bup, isf, isb, iso in zip(TO.fnspace_.pnames, x_aligned, b_dn, b_up, isfixed, isbound, isoutside):

        if isb and isf:
            comment = "FIX & ONBOUND"
        elif isb and not isf:
            comment="ONBOUND"
        elif not isb and isf:
            comment="FIX"
        elif iso and not isf:
            comment = "OUTSIDE"
        elif iso and isf:
            comment = "FIX & OUTSIDE"
        else:
            comment = ""
        s+= ("{:<{slen}}\t{:<{plen}} # {:<13} [ {:<{dnlen}}  ...  {:<{uplen}} ]\n".format(pn, val, comment, bdn, bup, slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
    return s

def writeResult(TO, x, fname, meta=None):
    with open(fname, "w") as f:
        if meta is not None:
           f.write("{}".format(meta))
        f.write("{}".format(printParams(TO,x)))

if __name__ == "__main__":
    import optparse, os, sys, h5py

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTDIR", default="tune", help="Output directory (default: %default)")
    op.add_option("-e", "--errorapprox", dest="ERRAPP", default=None, help="Approximations of bin uncertainties (default: %default)")
    op.add_option("-s", dest="SEED", type=int, default=1234, help="Random seed (default: %default)")
    op.add_option("-r", "--restart", dest="RESTART", default=1, type=int, help="Minimiser restarts (default: %default)")
    op.add_option("--msp", dest="MSP", default=None, help="Manual startpoint, comma separated string (default: %default)")
    op.add_option("-a", "--algorithm", dest="ALGO", default="tnc", help="The minimisation algorithm tnc, ncg, lbfgsb, trust (default: %default)")
    op.add_option("-l", "--limits", dest="LIMITS", default=None, help="Parameter file with limits and fixed parameters (default: %default)")
    op.add_option("-f", dest="FORCE", default=False, action = 'store_true', help="Overwrite output directory (default: %default)")
    op.add_option("-p", "--plotvalley", dest="PLOTVALLEY", default=False, action = 'store_true', help="Parameter dependence near minimum (default: %default)")
    op.add_option("--mode", dest="MODE", default="sip", help="Base algorithm  --- la |sip|lasip --- (default: %default)")
    op.add_option("--log", dest="ISLOG", action='store_true', default=False, help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("--ftol", dest="FTOL", type=float, default=1e-9, help="ftol for SLSQP (default: %default)")
    op.add_option("--rparam", dest="RPARAM", type=float, default=0, help="error on theory error (default: %default)")
    op.add_option("--therr",  dest="THERR", type=float, default=0, help="error on theory, currently a percentage of prediction (default: %default)")
    opts,args = op.parse_args() 

    if opts.ALGO not in ["tnc", "ncg", "lbfgsb" ,"trust"]:
        raise Exception("Minimisation algorithm {} not implemented, should be tnc, ncg, lbfgsb or trust, exiting".format(opts.ALGO))
    WFILE  = args[0]
    DATA   = args[1]
    APPROX = args[2]

    #TODO add protections and checks if files exist etc
    if not os.path.exists(opts.OUTDIR): os.makedirs(opts.OUTDIR)

    np.random.seed(opts.SEED)

    rank=0
    size=1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception as e:
        print("Exception when trying to import mpi4py:", e)
        comm = None
        pass

    import json

    with open(APPROX) as f: rd = json.load(f)
    blows = rd["__xmin"]
    rd.pop('__xmin',None)
    bups  = rd["__xmax"]
    rd.pop('__xmax',None)
    if '__vmin' in rd.keys():
        vlows = rd["__vmin"]
        rd.pop('__vmin',None)
        vups  = rd["__vmax"]
        rd.pop('__vmax',None)

    if opts.ERRAPP != None:
        with open(opts.ERRAPP) as f: rde = json.load(f)
        EAPP = []
    else:
        EAPP = None


    binids = TOOLS.sorted_nicely( rd.keys() )

    blows = np.array(blows)
    bups = np.array(bups)

    hnames  = [    b.split("#")[0]  for b in binids]
    bnums   = [int(b.split("#")[1]) for b in binids]

    matchers = read_pointmatchers(WFILE)
    weights = []
    for hn, bnum, blow, bup in zip(hnames, bnums, blows, bups):
        pathmatch_matchers = [(m, wstr) for  m, wstr  in matchers.items()    if m.match_path(hn)]
        posmatch_matchers  = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum, blow, bup)]
        w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
        weights.append(w)
    weights = np.array(weights)
    binids = np.array(binids)

    nonzero = np.nonzero( weights )

    dexp = IO.readExpData(DATA,binids[nonzero])

    """
    Is dexp guaranteed to be ordered in the right way?   This was the issue before.
    """

    DATA = np.array([dexp[b][0] for b in binids[nonzero]], dtype=np.float64)
    ERRS = np.array([dexp[b][1] for b in binids[nonzero]], dtype=np.float64)

    # delay until later?
    good = np.nonzero(ERRS)

    APPR = []
    for b in binids:
        P_output = rd[b]
        P_from_data_structure = PolynomialApproximation.from_data_structure(P_output)
        APPR.append(P_from_data_structure)
        if opts.ERRAPP != None:
            PE_from_data_structure = PolynomialApproximation.from_data_structure(rde[b])
            EAPP.append(PE_from_data_structure)


    # Do envelope filtering
    ev = (DATA > vlows) & (DATA < vups) & (ERRS > 0)

    good = np.nonzero( ERRS * ev)

    # Hypothesis filtering

    # Any bins left?

    """
    WGT  = weights[nonzero]
    BLOW = blows[nonzero]
    BUP  = bups[nonzero]
    BINS = binids[nonzero]
    """
    AP   = APPR
    EA   = EAPP
    WGT  = weights[good]
    BLOW = blows[good]
    BUP  = bups[good]
    BINS = binids[good]
    DATA = DATA[good]
    ERRS = ERRS[good]
    APPR = [APPR[g] for g in good[0].flatten()]
    if opts.ERRAPP: EAPP = [EAPP[g] for g in good[0].flatten()]

    ps = PolySet.from_surrogates(APPR)
    eps = PolySet.from_surrogates(EAPP) if opts.ERRAPP else None

    DIM = ps.fnspace_.dim
    PNAMES = ps.fnspace_.pnames

    fixed = []
    bounds = ps.fnspace_.box.T
    if opts.LIMITS is not None:
        from apprentice.io import read_limitsandfixed
        # check that opts.LIMITS is a legitimate file
        lim, fix = read_limitsandfixed(opts.LIMITS)
        # fixed parameters not working
        fixed = [ [PNAMES.index(k), fix[k]] for k in fix.keys()]
        lims  = [ [PNAMES.index(k), list(lim[k])] for k in lim.keys()]
        for i, b in lims:
            bounds[0][i] = b[0]
            bounds[1][i] = b[1]

    GG = TheoryErrorTuning(DIM, ps.fnspace_, DATA, ERRS, ps, eps, WGT, BINS, BLOW, BUP, bounds=bounds, fixed=fixed,
                           RPARAM=opts.RPARAM, THERR=opts.THERR)
    SC = ScipyMinimizer(GG,method=opts.ALGO)

    box = APPR[0].fnspace.box
    x0 = []
    if opts.MSP is not None:
        x0 = [float(x) for x in opts.MSP.split(",")]
    else:
        for i,b in enumerate(box):
            x = np.random.uniform(b[0],b[1])
            x0.append(x)

    x0 = np.array(x0)

    if opts.RESTART > 1 : x0 = None

    t0 = time.time()
    res = SC.minimize(x0, method = opts.ALGO, nrestart = opts.RESTART )
    t1 = time.time()
    print(res)


    x0 = res.x
#    GG.setWeights(dict(zip(binids[good],wt_flat)))
    chi2 = GG.objective(res.x, unbiased = True)
    ndf = len(WGT) - len(x0.flatten()) + 1

    meta  = "# Objective value at best fit point: %.2f (%.2f without weights)\n"%(res.fun, chi2)
    meta += "# Degrees of freedom: {}\n".format(ndf)
    meta += "# phi2/ndf: %.3f\n"%(chi2/ndf)
    meta += "# Minimisation took {} seconds\n".format(t1-t0)
    meta += "# Command line: {}\n".format(" ".join(sys.argv))
    meta += "# Best fit point:"

    print(meta)
    print(res.x)
    print(printParams(GG, res.x))

    DX = (bups - blows)*0.5
    X  = blows + DX
    Y  = [s.f_x(res.x) for s in AP]
    if EA==None:
        dY = np.zeros_like(Y)
    else:
        dY = [s.f_x(res.x) for s in EA]

    import yoda

    observables = np.unique( hnames )

    Y2D = list()
    for obs in observables:
        idx = np.where( np.array(hnames) == obs )
        try:
            P2D = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]
        except:
            P2D = [yoda.Point2D(x,y,dx,dy, source=b'') for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]

        Y2D.append(yoda.Scatter2D(P2D, obs, obs))

    outcommon = "{}_{}_{}".format(opts.ALGO, opts.RESTART, opts.SEED)
    fileNameYoda = os.path.join(opts.OUTDIR,"predictions_{}.yoda".format(outcommon))
    writeResult(GG, res.x, os.path.join(opts.OUTDIR, "minimum_{}.txt".format(outcommon)), meta=meta)
    mkPlotsCorrelation(GG, res.x, opts.OUTDIR+"/{}_".format(outcommon))
    if opts.PLOTVALLEY:
        plotout = os.path.join(opts.OUTDIR, "valleys_{}".format(outcommon))
        if not os.path.exists(plotout): os.makedirs(plotout)
        mkPlotsMinimum(GG, res.x, prefix=plotout+"/")
    yoda.write(Y2D,fileNameYoda)

