import inspect
import numpy as np

class Util(object):

    @staticmethod
    def inherits_from(child, parent_name):
        if inspect.isclass(child.__class__):
            if parent_name in [c.__name__ for c in inspect.getmro(child.__class__)[1:]]:
                return True
        return False


    @staticmethod
    def num_coeffs_poly(dim, order):
        """
        Number of coefficients a dim-dimensional polynomial of order order has.
        """
        ntok = 1
        r = min(order, dim)
        for i in range(r):
            ntok = ntok * (dim + order - i) / (i + 1)
        return int(ntok)

    @staticmethod
    def num_coeffs_to_order(dim, ncoeffs):
        """
        Infer the polynomial order from the number of coefficients and the dimension
        """
        if ncoeffs == 1:
            return 0
        curr  = 1
        order = -1
        while (curr !=ncoeffs):
            order +=1
            curr = Util.num_coeffs_poly(dim, order)
        return order

    @staticmethod
    def num_coeffs_rapp(dim, order):
        """
        Number of coefficients a dim-dimensional rational approximation of order (m,n) has.
        """
        return Util.num_coeffs_poly(dim, order[0]) + Util.num_coeffs_poly(dim, order[1])

    @staticmethod
    def gradient_recurrence(X, struct, jacfac, NNZ, sred):
        """
        X ... scaled point
        struct ... polynomial structure
        jacfac ... jacobian factor
        NNZ  ... list of np.where results
        sred ... reduced structure
        returns array suitable for multiplication with coefficient vector
        """
        # import numpy as np
        dim = len(X)
        REC = np.zeros((dim, len(struct)))
        _RR = np.power(X, struct)
        nelem = len(sred[0])

        W=[_RR[nz] for nz in NNZ]

        for coord, (RR, nz) in enumerate(zip(W,NNZ)):
            RR[:, coord] = jacfac[coord] * sred[coord] *_RR[:nelem, coord]
            REC[coord][nz] = np.prod(RR, axis=1)

        return REC


    @staticmethod
    def prime(GREC, COEFF, dim, NNZ):
        ret = np.zeros((len(COEFF), dim))
        for i in range(dim):
            for j in range(len(COEFF)):
                for k in NNZ[i]:
                    ret[j][i] += COEFF[j][k] * GREC[i][k]

        return ret

    # TODO jit here causes problems with oneAPI python
    # @jit(forceobj=True)#, parallel=True)
    @staticmethod
    def doubleprime(dim, xs, NSEL, HH, HNONZ, EE, COEFF):
        ret = np.zeros((dim, dim, NSEL), dtype=np.float64)
        for numx in range(dim):
            for numy in range(dim):
                rec = HH[numx][numy][HNONZ[numx][numy]] * Util.hreduction(xs, EE[numx][numy][HNONZ[numx][numy]])
                if numy>=numx:
                    ret[numx][numy] = np.sum((rec*COEFF[:,HNONZ[numx][numy]]), axis=1)
                else:
                    ret[numx][numy] = ret[numy][numx]

        return ret

    @staticmethod
    def hreduction(xs, ee):
        dim = len(xs)
        nel = len(ee)
        ret = np.ones(nel)
        for n in range(nel):
            for d in range(dim):
                if ee[n][d] == 0: continue
                if ee[n][d] == 1:
                    ret[n] *= xs[d]
                else:
                    ret[n] *= pow(xs[d], ee[n][d])
        return ret

    @staticmethod
    def calcSpans(spans1, DIM, G1, G2, H2, H3, grads, egrads):
        for numx in range(DIM):
            for numy in range(DIM):
                if numy<=numx:
                    spans1[numx][numy] +=        G1 *  grads[:,numx] *  grads[:,numy]
                    spans1[numx][numy] +=        G2 * (egrads[:,numx] *  grads[:,numy] + egrads[:,numy] *  grads[:,numx])
                    spans1[numx][numy] += (H2 + H3) * egrads[:,numx] * egrads[:,numy]
        for numx in range(DIM):
            for numy in range(DIM):
                if numy>numx:
                    spans1[numx][numy] = spans1[numy][numx]
        return spans1
