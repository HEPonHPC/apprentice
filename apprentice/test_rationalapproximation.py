import unittest
import numpy as np
import pprint

import apprentice
from apprentice.rationalapproximation import RationalApproximation

class TestRationalApproximation(unittest.TestCase):
    @staticmethod
    def get_data_1D(scaled=False):
        (X,_,_) = TestRationalApproximation.get_data_2D(scaled)
        X_1D = [[x[0]] for x in X]
        (Y_1D_21,Y_1D_22) = TestRationalApproximation.get_y_vals_1D(X_1D)

        return X_1D,Y_1D_21,Y_1D_22

    @staticmethod
    def get_test_data_1D(scaled=False):
        (X,_,_) = TestRationalApproximation.get_test_data_2D(scaled)
        X_1D = [[x[0]] for x in X]
        (Y_1D_21,Y_1D_22) = TestRationalApproximation.get_y_vals_1D(X_1D)

        return X_1D,Y_1D_21,Y_1D_22

    @staticmethod
    def get_y_vals_1D(X):
        Y_1D_21 = [(x[0]**2 + 3*x[0] )/(2*x[0]+1)for x in X]
        Y_1D_22 = [(x[0]**2 + 2*x[0])/(2*x[0]**2+4) for x in X]
        return Y_1D_21,Y_1D_22

    @staticmethod
    def get_y_vals_2D(X):
        Y_2D_21 = [(x[0]**2 + 3*x[0]*x[1] + x[1]**2)/(7*x[0]+3*x[1])for x in X]
        Y_2D_22 = [(x[0]**2 + 2*x[0]*x[1] + x[1]**2)/(2*(x[0]**2)+2*x[1]+4) for x in X]
        return Y_2D_21,Y_2D_22

    @staticmethod
    def get_test_data_2D(scaled=False):
        X_2D_unscaled = [[0.3337290751538484, 27.51274282370659],
                         [0.6540702777506159, 26.3872070199262],
                         [0.6855372636713173, 21.79260623614056],
                         [0.6398524260206057, 36.57463051282524],
                         [0.5283760780191591, 29.917568182408793],
                         [0.7715776309137791, 28.220143862544376],
                         [0.528199809315267, 33.04825216135785],
                         [0.0850724447805291, 35.73749495122373],
                         [0.6075833315374508, 36.96759089087896],
                         [0.2865528097461917, 34.7440219659164],
                         [0.8750368015095307, 28.694717258230753],
                         [0.5051214296708091, 29.432085831056284],
                         [0.1499077214853949, 23.728262330615387],
                         [0.3557725926866787, 24.98334055787932],
                         [0.22186564915193632, 30.827015359151734],
                         [0.10277047821881302, 29.23206655759604],
                         [0.1926825137364244, 25.366343192331364],
                         [0.2989129196965349, 32.03116823073295],
                         [0.18067444304164437, 25.86372489492742],
                         [0.5935004097716723, 30.062459400024686]]

        X_2D_scaled = [[ -1.,-1.],
                       [-0.37046114, -0.24610974],
                       [ 0.44056584, -0.39445068],
                       [ 0.52023269, -1.        ],
                       [ 0.40456965,  0.94820945],
                       [ 0.12233832,  0.07083627],
                       [ 0.73806623, -0.15287722],
                       [ 0.12189205,  0.48344742],
                       [-1.        ,  0.83787846],
                       [ 0.32287206,  1.        ],
                       [-0.48990011,  0.70694284],
                       [ 1.        , -0.09033041],
                       [ 0.06346313,  0.00685171],
                       [-0.83585265, -0.74488856],
                       [-0.31465225, -0.57947446],
                       [-0.65367246,  0.19069763],
                       [-0.95519283, -0.01951   ],
                       [-0.72755715, -0.5289963 ],
                       [-0.45860728,  0.34939998],
                       [-0.7579587 , -0.46344346],
                       [ 0.28721748,  0.08993233],
                       [ 1.,1.]]

        if scaled:
            (Y_2D_21_scaled,Y_2D_22_scaled) = TestRationalApproximation.get_y_vals_2D(X_2D_scaled)
            return X_2D_scaled,Y_2D_21_scaled,Y_2D_22_scaled
        else:
            (Y_2D_21_unscaled,Y_2D_22_unscaled) = TestRationalApproximation.get_y_vals_2D(X_2D_unscaled)
            return X_2D_unscaled,Y_2D_21_unscaled,Y_2D_22_unscaled

    @staticmethod
    def get_data_2D(scaled=False):
        X_2D_unscaled = [[5.88130801e-01, 3.71525106e+01],
                         [8.97713728e-01, 3.89955805e+01],
                         [8.91530729e-01, 3.12337372e+01],
                         [8.15837477e-01, 2.35756104e+01],
                         [3.58895856e-02, 3.54050387e+01],
                         [6.91757582e-01, 2.98476208e+01],
                         [3.78680942e-01, 3.26250613e+01],
                         [5.18510945e-01, 3.67899585e+01],
                         [6.57951466e-01, 2.92207879e+01],
                         [1.93850218e-01, 2.99588015e+01],
                         [2.72316402e-01, 3.35882224e+01],
                         [7.18605934e-01, 3.30157183e+01],
                         [7.83003609e-01, 2.53759048e+01],
                         [8.50327640e-01, 2.13464933e+01],
                         [7.75244894e-01, 3.54289028e+01],
                         [3.66643064e-02, 2.96196826e+01],
                         [1.16693735e-01, 2.65841282e+01],
                         [7.51280699e-01, 3.02128211e+01],
                         [2.39218216e-01, 2.52725766e+01],
                         [2.54806014e-01, 2.62102310e+01]]
        X_2D_scaled = [[ -1.,-1.],
                       [ 0.1762616 ,  0.71525106],
                       [ 0.79542746,  0.89955805],
                       [ 0.78306146,  0.12337372],
                       [ 0.63167495, -0.64243896],
                       [-0.92822083,  0.54050387],
                       [ 0.38351516, -0.01523792],
                       [-0.24263812,  0.26250613],
                       [ 0.03702189,  0.67899585],
                       [ 0.31590293, -0.07792121],
                       [-0.61229956, -0.00411985],
                       [-0.4553672 ,  0.35882224],
                       [ 0.43721187,  0.30157183],
                       [ 0.56600722, -0.46240952],
                       [ 0.70065528, -0.86535067],
                       [ 0.55048979,  0.54289028],
                       [-0.92667139, -0.03803174],
                       [-0.76661253, -0.34158718],
                       [ 0.5025614 ,  0.02128211],
                       [-0.52156357, -0.47274234],
                       [-0.49038797, -0.3789769 ],
                       [1.,1.]]


        if scaled:
            (Y_2D_21_scaled,Y_2D_22_scaled) = TestRationalApproximation.get_y_vals_2D(X_2D_scaled)
            return X_2D_scaled,Y_2D_21_scaled,Y_2D_22_scaled
        else:
            (Y_2D_21_unscaled,Y_2D_22_unscaled) = TestRationalApproximation.get_y_vals_2D(X_2D_unscaled)
            return X_2D_unscaled,Y_2D_21_unscaled,Y_2D_22_unscaled

    @staticmethod
    def get_exp_pcoeff(order=(2,1)):
        return [0.,0.,0.,1.,2.,1.]

    @staticmethod
    def get_exp_qcoeff(order=(2,1)):
        if order[1] == 1:
            return [0.,2.,2.]
        elif order[1] == 2:
            return [0.,0.,0.,2.,0.,3.]

    def test_from_interpolation_points_s11_unscaled_ampl_useabsmodel_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            use_abstract_model=True,
                                                            fit_solver='ipopt',
                                                            strategy="1.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s21_unscaled_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s22_unscaled_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s23_unscaled_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.3")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_unscaled_fsscipy_lsscipy_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='scipy',
                                                            local_solver='scipy',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_unscaled_fsampl_lsampl_1D(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_1D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='ipopt',
                                                            local_solver='ipopt',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_1D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s11_unscaled_ampl_useabsmodel(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                                m=2,
                                                                n=1,
                                                                pnames=['x','y'],
                                                                use_abstract_model=True,
                                                                fit_solver='ipopt',
                                                                strategy="1.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)


    def test_from_interpolation_points_s11_unscaled_ampl_dontuseabsmodel(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            use_abstract_model=False,
                                                            fit_solver='ipopt',
                                                            strategy="1.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s11_scaled_ampl_dontuseabsmodel(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            use_abstract_model=False,
                                                            fit_solver='ipopt',
                                                            strategy="1.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-1)

    def test_from_interpolation_points_s11_scaled_scipy(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            fit_solver='scipy',
                                                            strategy="1.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-1)

    def test_from_interpolation_points_s21_unscaled(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s21_scaled(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s22_unscaled(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s22_scaled(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s23_unscaled(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.3")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s23_scaled(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.3")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y21_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y21_T_R,Y21_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_unscaled_fsscipy_lsscipy(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='scipy',
                                                            local_solver='scipy',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_scaled_fsscipy_lsscipy(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='scipy',
                                                            local_solver='scipy',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_unscaled_fsampl_lsampl(self):
        scaled = False
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='ipopt',
                                                            local_solver='ipopt',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_from_interpolation_points_s31_scaled_fsampl_lsampl(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y22,
                                                            m=2,
                                                            n=2,
                                                            fit_solver='ipopt',
                                                            local_solver='ipopt',
                                                            pnames=['x','y'],
                                                            strategy="3.1")
        (X_T,Y21_T,Y22_T) = TestRationalApproximation.get_test_data_2D(scaled=scaled)
        Y22_T_R = R.f_X(X_T)
        C2 = sum([(y1-y2)**2 for (y1,y2) in zip(Y22_T_R,Y22_T)])
        assert(C2 < 1e-6)

    def test_save_and_from_file(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        tmp_file = "/tmp/pa.json"
        R.save(tmp_file)
        R_from_file = RationalApproximation.from_file(tmp_file)
        assert(np.all(np.isclose(R_from_file.coeff_numerator, R.coeff_numerator))
            and np.all(np.isclose(R_from_file.coeff_denominator, R.coeff_denominator)))

    def test_gradient(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        exp_grad = [0.15, 0.35]
        grad = R.gradient([1,1])
        R.save("/tmp/gradtest.json")
        assert(np.all(np.isclose(grad, exp_grad)))

    def hessian(self, X):
        # Hessian not implemeted
        assert(True)

    def test_coeff_norm(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        assert (np.isclose(R.coeff_norm, 13552146108353.836))

    def test_coeff2_norm(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['x','y'],
                                                            strategy="2.2")
        assert (np.isclose(R.coeff2_norm, 7504838683826.167))

    def test_set_pnames(self):
        scaled = True
        (X,Y21,Y22) = TestRationalApproximation.get_data_2D(scaled=scaled)
        R = RationalApproximation.from_interpolation_points(X,Y21,
                                                            m=2,
                                                            n=1,
                                                            pnames=['dim1','dim2'],
                                                            strategy="2.2")
        expected_pnames = ["dim1","dim2"]
        given_pnames = R.pnames
        assert (given_pnames==expected_pnames)

if __name__ == "__main__":
    unittest.main()
