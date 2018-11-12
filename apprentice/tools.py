def numCoeffsPoly(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    ntok = 1
    r = min(order, dim)
    for i in range(r):
      ntok = ntok*(dim+order-i)/(i+1)
    return int(ntok)

def numCoeffsRapp(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    return 1 + numCoeffsPoly(dim, order[0]) + numCoeffsPoly(dim, order[1])

def maxOrder(N, dim):
    """
    Utility function to find highest order polynomial
    for dataset with N dim-dimensional points.
    Intended to be used when constructing max ONB
    """
    from scipy.special import comb
    omax = 0
    while comb(dim + omax+1, omax+1) + 1 <= N: # The '+1' stands for a order 0 polynomial's dof
        omax+=1
    return omax
