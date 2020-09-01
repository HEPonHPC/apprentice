
def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

@timeit
def cholesky(mat):
    import numpy as np
    return np.linalg.cholesky(mat)


@timeit
def outerprod(X,u):
    import numpy as np
    return X + np.outer(u, u)

@timeit
def cholupdate(X, u):
    # cholupdate https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
    import choldate
    return choldate.cholupdate(X, u)

@timeit
def choldowndate(X,u):
    # choldowndate https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_downdate
    import choldate
    return choldate.choldowndate(X, u)
@timeit
def testR1updatedowndate():
    import numpy as np
    from sklearn.datasets import make_spd_matrix

    np.random.seed(1)

    K = make_spd_matrix(n_dim=2,random_state=23242)
    sigma = np.array([2.] * K.shape[0])

    R = cholesky(K)

    Rtilde = R.copy()
    cholupdate(Rtilde, sigma)

    res1 = np.matmul(Rtilde,Rtilde.transpose())

    sigmaI = np.zeros((R.shape[0], R.shape[0]))
    np.fill_diagonal(sigmaI, sigma)

    res2 =  np.matmul(R,R.transpose()) + sigmaI
    print(np.all((res1 - res2) ** 2 < 1e-16))

    exit(1)


    # Create a random update vector, u
    u = np.array([2.]*R.shape[0])
    # Calculate the updated positive definite matrix, V1, and its Cholesky factor, R1
    u1 = np.zeros((R.shape[0], R.shape[0]))
    np.fill_diagonal(u1, u)
    # X1 = K + u1
    print(K)
    X1 = outerprod(K,u)
    print(X1)
    R1 = X1
    exit(1)
    # The following is equivalent to the above
    R1_ = R.copy()
    for i in range(len(u)):
        cholupdate(R1_, u.copy())
    res1_ = np.matmul(R1_,R1_.transpose())

    print((R1 - res1_) ** 2)
    print(np.all((R1 - res1_) ** 2 < 1e-16))
    exit(1)
    # And downdating is the inverse of updating
    R_ = R1.copy()
    choldowndate(R_, u.copy())
    print(np.all((R - R_) ** 2 < 1e-16))

if __name__ == "__main__":
    testR1updatedowndate()
