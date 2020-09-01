
import numpy as np
from sklearn.datasets import make_spd_matrix
import chol_update
from timeit import default_timer as timer


def cholesky(mat):
    return np.linalg.cholesky(mat)


def test_diagonal_update(dimension=5, random_seed=23242):
    """
    Test Cholesky factorization of K + diag(d), via rank one update
    """
    print("\n***\nTesting Cholesky Rank-One-Update: Diagonal Matrix Update\n***")
    # Create SPD toy matrix
    K = make_spd_matrix(n_dim=dimension, random_state=random_seed)

    # update vector x, such that the new matrix is K + np.outer(x, x)
    sigma = 2.0 * np.ones(dimension)  # this results in K + 2*I

    # Updated matrix: K + x xT
    updated_K = K + np.diag(sigma)

    print("Original Matrix K: \n %s" % repr(K))
    print("Updated Matrix K+diag(x): \n %s" % repr(updated_K))

    # Cholesky factor (lower trianglular) of K
    R = cholesky(K)
    print("Exact Cholesky of K: \n %s" % repr(R))

    # Cholesky factor (lower trianglular) of updated K
    updated_R = cholesky(updated_K)
    print("Exact Cholesky of K + diag(x): \n %s" % repr(updated_R))

    # Cholesky Update (should be same as updated_R)
    Rtilde = chol_update.chol_diag_update(R, sigma, in_place=False)
    print("Rank-One update of Cholesky >> K + diag(x): \n %s" % repr(Rtilde))

    # Merge for comparison
    Ktilde = np.dot(Rtilde, Rtilde.T)
    print("Retrieved version of updated K: K+diag(x): \n %s" % repr(Ktilde))

    # Compare exact updated cholesky to rank-one-update
    print("\nCholesky Update Maches Exact Cholesky: %s" % np.allclose(updated_R, Rtilde))
    print("Matrix K+diag(x) matches result from Rank-One-Update: %s\n" % np.allclose(updated_K, Ktilde))

def test_mean_gp_prediction(dimension=5, random_seed=23242, usecholdate = True):
    # We assume 0 mean for this function
    # \overline{y_i(p)} = K(p,P)[K + \sigma^2]\inv(y)
    # meangpy           = KpP [K + sigma^2]\inv(y)

    K = make_spd_matrix(n_dim=dimension, random_state=random_seed)
    KpP = 1.0 * np.ones(dimension)
    y = 4.0 * np.ones(dimension)
    sigma2 = 2.0 * np.ones(dimension)

    # Cholesky factor (lower trianglular) of K
    R = cholesky(K)

    """
    Prediction using conventional method
    """
    st = timer()
    # Updated matrix: K + x xT
    Kpsigma2 = K + np.diag(sigma2)
    meangpy_conv = np.matmul(np.matmul(KpP,np.linalg.inv(Kpsigma2)),y)
    print("GP mean from Conventional Mehod= {}".format(meangpy_conv))
    print("Took {} seconds".format(timer()-st))

    """
    Prediction using rank 1 update method
    """
    st = timer()
    from scipy.linalg import solve_triangular
    # Cholesky Update (should be same as updated_R)
    Rtilde = chol_update.chol_diag_update(R, sigma2, in_place=False,usecholdate=usecholdate)
    Ktilde = np.dot(Rtilde, Rtilde.T)
    print(np.all((Kpsigma2 - Ktilde) ** 2 < 1e-16))
    print("Done with chol_diag_update in {}".format(timer()-st))
    z = solve_triangular(Rtilde,y,lower=True)
    print("Done with solve triangular 1 in {}".format(timer()-st))
    x = solve_triangular(Rtilde,z,lower=True, trans='T')
    print("Done with solve triangular 2 in {}".format(timer()-st))
    meangpy_r1up = np.matmul(KpP,x)

    print("\nGP mean from Rank 1 update = {}\n".format(meangpy_r1up))
    print("Took {} seconds".format(timer() - st))

    """
    Using cholupdate from choldate
    """

def test_full_matrix_update(dimension=3, random_seed=23242):
    """
    """
    print("\n***\nTesting Cholesky Rank-One-Update: Full Matrix Update\n***")
    # Create SPD toy matrix
    K = make_spd_matrix(n_dim=dimension, random_state=random_seed)

    # update vector x, such that the new matrix is K + np.outer(x, x)
    sigma = 2.0 * np.ones(dimension)  # this results in K + [[4, 4],[4,4]]

    # Updated matrix: K + x xT
    updated_K = K + np.outer(sigma, sigma)

    print("Original Matrix K: \n %s" % repr(K))
    print("Updated Matrix K+diag(x): \n %s" % repr(updated_K))

    # Cholesky factor (lower trianglular) of K
    R = cholesky(K)
    print("Exact Cholesky of K: \n %s" % repr(R))

    # Cholesky factor (lower trianglular) of updated K
    updated_R = cholesky(updated_K)
    print("Exact Cholesky of K + xx*: \n %s" % repr(updated_R))

    # Cholesky Update (should be same as updated_R)
    Rtilde, _ = chol_update.chol_update(R, sigma, in_place=False)
    print("Rank-One update of Cholesky >> K + xx*: \n %s" % repr(Rtilde))

    # Merge for comparison
    Ktilde = np.dot(Rtilde, Rtilde.T)
    print("Retrieved version of updated K: K+diag(x): \n %s" % repr(Ktilde))

    # Compare exact updated cholesky to rank-one-update
    print("\nCholesky Update Maches Exact Cholesky: %s" % np.allclose(updated_R, Rtilde))
    print("Matrix K+xxT matches result from Rank-One-Update: %s\n" % np.allclose(updated_K, Ktilde))


def test_chol_update(dimension=3, random_seed=23242):
    from choldate import cholupdate, choldowndate
    K = make_spd_matrix(n_dim=dimension, random_state=random_seed)
    RL = cholesky(K)
    RR = RL.transpose()

    sigma2 = 2.0 * np.ones(dimension)

    updated_K = K + np.outer(sigma2, sigma2)
    updated_RL = cholesky(updated_K)
    updated_RR = updated_RL.transpose()

    RR_notP = RR.copy()
    RL_P = RL.copy()
    chol_update.cholupdate(RR_notP, sigma2.copy())
    chol_update.chol_update(RL_P,sigma2,in_place = True)

    print(np.all((RR_notP.transpose() - RL_P)**2 < 1e-16))
    print(np.all((RR_notP - updated_RR) ** 2 < 1e-16))
    print(np.all((RL_P - updated_RL) ** 2 < 1e-16))






if __name__ == "__main__":
    np.random.seed(1)
    dimension = 4000
    # test_full_matrix_update()
    # test_diagonal_update()
    test_mean_gp_prediction(dimension=dimension,usecholdate=True)
    test_mean_gp_prediction(dimension=dimension, usecholdate=False)
    # test_chol_update(dimension=4)

