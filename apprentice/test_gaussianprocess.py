import unittest
import numpy as np
import pprint
import json
import pandas as pd
from io import StringIO

from apprentice.gaussianprocess import GaussianProcess
from apprentice.rationalapproximation import RationalApproximation

class TestGaussianProcess(unittest.TestCase):

    @staticmethod
    def get_available_kernels():
        return  ["sqe","ratquad","matern32","matern52","poly"]

    def test_from_interpolation_points_gp(self):
        data = pd.read_csv(StringIO(TestGaussianProcess.get_data()),header=None)
        D = data.values
        X = D[:, :-2]
        Y = D[:,-2:]
        mean_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_mean_surrogate_model()))
        err_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_error_surrogate_model()))
        ans = True
        for k in TestGaussianProcess.get_available_kernels():
            GP = GaussianProcess.from_interpolation_points(X,Y,
                                                           seed=32434,
                                                           kernel=k,
                                                           max_restarts=2,
                                                           keepout_percentage=20.0,
                                                           mean_surrogate_model=mean_sm,
                                                           error_surrogate_model=err_sm,
                                                           sample_size=25,
                                                           stopping_bound=10**-2,
                                                            strategy="HoGP"
                                                           )
            (m,s) = GP([0.2680424119976856,
                        1.9903768915597309,
                        0.1794978694693918])
            filepath = '/tmp/gp.json'
            GP.save(filepath)
            GP_from_file = GaussianProcess.from_file(filepath,
                                                     mean_surrogate_model=mean_sm,
                                                     error_surrogate_model=err_sm
                                                     )
            (m_t,s_t) = GP_from_file([0.2680424119976856,
                          1.9903768915597309,
                          0.1794978694693918])

            ans = ans and np.isclose(np.array(m,s),np.array(m_t,s_t),rtol=1e-2)
        assert(ans)

    def test_from_interpolation_points_mlhgp(self):
        data = pd.read_csv(StringIO(TestGaussianProcess.get_data()),header=None)
        D = data.values
        X = D[:, :-2]
        Y = D[:,-2:]
        mean_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_mean_surrogate_model()))
        err_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_error_surrogate_model()))
        ans = True
        for k in TestGaussianProcess.get_available_kernels():
            GP = GaussianProcess.from_interpolation_points(X,Y,
                                                           seed=32434,
                                                           kernel=k,
                                                           max_restarts=2,
                                                           keepout_percentage=20.0,
                                                           mean_surrogate_model=mean_sm,
                                                           error_surrogate_model=err_sm,
                                                           sample_size=25,
                                                           stopping_bound=10**-4,
                                                           strategy="HeGP-ML"
                                                           )
            (m,s) = GP([0.2680424119976856,
                        1.9903768915597309,
                        0.1794978694693918])
            filepath = '/tmp/gp-mlhgp.json'
            GP.save(filepath)
            GP_from_file = GaussianProcess.from_file(filepath,
                                                     mean_surrogate_model=mean_sm,
                                                     error_surrogate_model=err_sm
                                                     )
            (m_t,s_t) = GP_from_file([0.2680424119976856,
                                      1.9903768915597309,
                                      0.1794978694693918])
            ans = ans and np.isclose(np.array(m,s),np.array(m_t,s_t),rtol=1e-2)
        assert(ans)

    def test_from_interpolation_points_sk(self):
        data = pd.read_csv(StringIO(TestGaussianProcess.get_data()),header=None)
        D = data.values
        X = D[:, :-2]
        Y = D[:,-2:]
        mean_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_mean_surrogate_model()))
        err_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_error_surrogate_model()))
        ans = True
        for k in TestGaussianProcess.get_available_kernels():
            GP = GaussianProcess.from_interpolation_points(X,Y,
                                                           seed=32434,
                                                           kernel=k,
                                                           max_restarts=2,
                                                           keepout_percentage=20.0,
                                                           mean_surrogate_model=mean_sm,
                                                           error_surrogate_model=err_sm,
                                                           sample_size=25,
                                                           stopping_bound=10**-4,
                                                           strategy="HeGP-SK"
                                                           )
            (m,s) = GP([0.2680424119976856,
                        1.9903768915597309,
                        0.1794978694693918])
            filepath = '/tmp/gp-sk.json'
            GP.save(filepath)
            GP_from_file = GaussianProcess.from_file(filepath,
                                                     mean_surrogate_model=mean_sm,
                                                     error_surrogate_model=err_sm
                                                     )
            (m_t,s_t) = GP_from_file([0.2680424119976856,
                                      1.9903768915597309,
                                      0.1794978694693918])
            ans = ans and np.isclose(np.array(m,s),np.array(m_t,s_t),rtol=1e-2)
        assert(ans)

    def test_mpi_tune(self):
        data = pd.read_csv(StringIO(TestGaussianProcess.get_data()),header=None)
        D = data.values
        X = D[:, :-2]
        Y = D[:,-2:]
        mean_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_mean_surrogate_model()))
        err_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_error_surrogate_model()))
        GP = GaussianProcess.from_interpolation_points(X,Y,
                                                           seed=32434,
                                                           kernel="sqe",
                                                           max_restarts=2,
                                                           keepout_percentage=20.0,
                                                           mean_surrogate_model=mean_sm,
                                                           error_surrogate_model=err_sm,
                                                           sample_size=25,
                                                           stopping_bound=10**-2,
                                                           strategy="HoGP",
                                                           use_mpi_tune=True
                                                           )
        (m,s) = GP([0.2680424119976856,
                    1.9903768915597309,
                    0.1794978694693918])
        filepath = '/tmp/gp.json'
        GP.save(filepath)
        GP_from_file = GaussianProcess.from_file(filepath,
                                                 mean_surrogate_model=mean_sm,
                                                 error_surrogate_model=err_sm
                                                 )
        (m_t,s_t) = GP_from_file([0.2680424119976856,
                                  1.9903768915597309,
                                  0.1794978694693918])

        assert(np.isclose(np.array(m,s),np.array(m_t,s_t),rtol=1e-2))

    def test_f_X(self):
        data = pd.read_csv(StringIO(TestGaussianProcess.get_data()),header=None)
        D = data.values
        X = D[:, :-2]
        Y = D[:,-2:]
        mean_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_mean_surrogate_model()))
        err_sm = RationalApproximation.from_data_structure(json.loads(TestGaussianProcess.get_error_surrogate_model()))
        GP = GaussianProcess.from_interpolation_points(X,Y,
                                                       seed=32434,
                                                       kernel="sqe",
                                                       max_restarts=2,
                                                       keepout_percentage=20.0,
                                                       mean_surrogate_model=mean_sm,
                                                       error_surrogate_model=err_sm,
                                                       sample_size=25,
                                                       stopping_bound=10**-2,
                                                       strategy="HeGP-ML"
                                                       )
        (m,s) = GP.f_X([[0.2680424119976856,
                    1.9903768915597309,
                    0.1794978694693918],
                    [0.2680424119976856,
                     1.9903768915597309,
                     0.1794978694693918]])
        filepath = '/tmp/gp.json'
        GP.save(filepath)
        GP_from_file = GaussianProcess.from_file(filepath,
                                                 mean_surrogate_model=mean_sm,
                                                 error_surrogate_model=err_sm
                                                 )
        (m_t,s_t) = GP_from_file.f_X([[0.2680424119976856,
                                  1.9903768915597309,
                                  0.1794978694693918],
                                  [0.2680424119976856,
                                   1.9903768915597309,
                                   0.1794978694693918]])

        assert(np.all(np.isclose(m,m_t)) and np.all(np.isclose(s,s_t)))

    @staticmethod
    def get_data():
        # From SimulationData/3D_miniapp/AvgData/ne100000_ns1/Bin5.csv
        s = """0.3830389007577846,1.3197957878716975,0.4377277390071145,18.552666666666667,0.11121350837215975
1.577460285881491,0.770305019903968,0.5680986526260692,20.169999999999998,0.1159597631364719
1.6443195771517811,1.330337117830091,0.1179230575007103,19.236666666666665,0.1132450636647993
1.7382547791224514,0.9851121630122228,0.8021476420801592,19.676000000000002,0.11453092740973214
0.28753364902912915,1.4676697480130036,0.7045813081895725,17.846666666666668,0.10907693513194153
0.4375842113481772,1.8647617315080167,0.44214075540417663,17.944000000000003,0.10937397618568444
1.8186319179449448,0.30765660100373343,0.18428708381381362,19.395333333333337,0.11371113499663181
0.0947105576030303,1.4147856984481944,0.5946247799344488,17.665333333333333,0.10852137526261309
1.0666203259975011,0.27798331285064637,0.5614330800633978,20.537333333333333,0.11701092066792551
0.65933689124183,1.1053402996027129,0.11189431757440382,18.638666666666666,0.11147097280358587
1.2143874124369691,1.2187003574909565,0.006764061990002789,18.55466666666667,0.11121950268625452
1.234883417608594,1.8418211955796768,0.7905241330570335,18.433333333333337,0.1108552609887726
1.5707171674275384,1.6039564546138465,0.2725926052826416,19.553333333333335,0.11417335746817449
1.984162932376724,1.92584317187516,0.7919641352916397,18.634,0.11145701712618487
0.5705019200490196,1.3248500695506396,0.4780937956706746,18.765333333333334,0.11184910470013706
0.3913503573317965,0.8881714136567117,0.05387368514623658,18.356666666666666,0.11062448995488196
0.903296816521718,1.9676085347395185,0.123942700486963,18.59,0.1113253490150978
0.2387617958524968,1.5293415010580245,0.5873036334639846,17.890666666666668,0.10921131402520122
0.9432650686407356,0.3928282709489593,0.2292185654606179,19.737333333333332,0.11470929440207633
1.7999303896733505,0.9501563680448478,0.5358516625316159,20.156000000000002,0.1159195123063125
0.012417033174258796,0.741155070386542,0.4368931721756102,18.354,0.11061645447219866
1.2242979941315149,1.8527565356850315,0.6257366699625353,18.577333333333335,0.1112874156806999
1.4119951301635465,0.4697006887818689,0.7460634091367166,20.411333333333335,0.11665142757615767
0.5529285102861933,1.643369919563035,0.9581393536837054,17.544,0.1081480466767662
1.6620139848670756,1.3407063841117624,0.4383098811224275,19.649333333333335,0.11445328984155743
0.3051455493490107,1.2231373074449423,0.5282242775850605,18.39333333333333,0.11073491871231145
1.9028575275071868,1.0646465213180287,0.5025595633825504,20.12666666666667,0.11583513188052136
1.0737563858488195,1.6745637207154846,0.0571156380888599,18.538666666666668,0.11117153912360445
1.3388434861490974,1.5808099310830501,0.7081153619776038,19.0,0.11254628677422755
1.5937343674503934,1.2039694911694092,0.9658365319921278,18.792666666666666,0.11193053401304062
0.29431379978599437,0.2533646009637481,0.5938934926247716,19.696666666666665,0.1145910603455222
0.22813139748532496,1.91145773015142,0.32570741442534723,17.962,0.10942882009172294
0.38723738030755445,1.0240609680153696,0.9204025710930878,17.865333333333332,0.10913396456750861
1.7581383230293517,0.6547083590837544,0.3480087928693462,20.384666666666668,0.11657520224206253
1.7518652694841894,0.84407108592416,0.5009951255234587,20.242,0.11616654710658603
0.36517746316061755,1.8232328924677856,0.7065281631717975,17.344,0.1075298408194984
1.4533169231242815,1.8201581062574737,0.7791638007693242,18.516,0.1111035552986492
1.198309561208585,0.7240254408161747,0.15139526440743212,19.468666666666667,0.11392590184462492
0.6703493182988698,1.383593198884075,0.07334254363261816,18.621333333333336,0.11141912861902226
0.1100127908124473,0.7817506650581176,0.5904818044629861,18.384,0.1107068200247844
1.7077971342512834,0.7167123650000161,0.1730672268147921,19.540000000000003,0.11413442367080437
0.26804241199768564,1.9903768915597309,0.17949786946939186,17.99866666666667,0.10954045422176736
0.6350936460543917,1.222924528386393,0.009348574500244507,18.547333333333334,0.11119752195480298
1.8012972423101836,1.9590345756606569,0.5568946791368349,19.302666666666667,0.1134391662718148
0.1695476867883803,0.7994044383123802,0.7284286763696719,18.316,0.11050188535344846
1.3669258703442724,1.4828636485692206,0.3702507547903949,19.293333333333333,0.11341173758576412
0.28487074668363443,1.1944440910955822,0.27304325968368554,18.48,0.11099549540409287
1.94899027617452,1.402016431061068,0.25565328643413965,19.508666666666667,0.11404287692696014
0.2166229883654576,1.5971253017129883,0.782477992600202,17.252000000000002,0.10724426946617398
1.5232078286149882,1.8459256038847824,0.6586227819424225,18.779333333333334,0.11189081980017644
1.1367351631458649,0.5631602464153609,0.6982963755517612,19.912,0.11521573966549305"""
        return s

    @staticmethod
    def get_mean_surrogate_model():
        # From SimulationData/3D_miniapp_RApp/AvgData/ne100000_ns1/Bin5.json
        s = """{
    "m": 3,
    "n": 1,
    "training_size": 100,
    "strategy": "2.2",
    "pcoeff": [
        3.86703063890504,
        -0.09009613429779662,
        -0.19814768708298836,
        0.20229890699098352,
        -0.17008119343696615,
        -0.11014960556865505,
        0.034207402829053324,
        0.058219746692483976,
        0.015954418627692868,
        -0.06210287565319317,
        0.07684507841605515,
        0.06247310618002777,
        0.06012024805632343,
        0.013385006160332114,
        -0.0701175490635324,
        -0.040701008848846865,
        -0.019466581748263545,
        0.010932358604808234,
        0.01354482351605757,
        0.005099469665261536
    ],
    "qcoeff": [
        0.20000000000000018,
        4.440892098500626e-16,
        -4.440892098500626e-16,
        -4.440892098500626e-16
    ],
    "fnspace": {
        "dim_": 3,
        "b_": [
            1.9841629323767243,
            1.9903768915597309,
            0.9935673628631738
        ],
        "a_": [
            0.0043788616462432355,
            0.2533646009637481,
            0.006764061990002789
        ],
        "sa_": [
            -1.0,
            -1.0,
            -1.0
        ],
        "sb_": [
            1.0,
            1.0,
            1.0
        ],
        "pnames_": [
            "x",
            "y",
            "z"
        ]
    }
}"""
        return s

    @staticmethod
    def get_error_surrogate_model():
        # From SimulationData/3D_miniapp_RApp/AvgData/ne100000_ns1/errBin5.json
        s ="""{
    "m": 3,
    "n": 1,
    "training_size": 100,
    "strategy": "2.2",
    "pcoeff": [
        0.022706934826678293,
        -0.0002671363563793161,
        -0.00058054263220364,
        0.0005946561211970236,
        -0.0005032723499938108,
        -0.00032807595378492316,
        9.486458282292887e-05,
        0.00017532156941668653,
        5.9267273765706285e-05,
        -0.00019092126969155743,
        0.00022846734331460894,
        0.00017338918328635433,
        0.0001689539193433509,
        3.9536649397153845e-05,
        -0.0001990981675661363,
        -0.00010899558399368736,
        -5.599922835353155e-05,
        2.8809025796272536e-05,
        3.311416848550408e-05,
        1.970017669183921e-05
    ],
    "qcoeff": [
        0.20000000000000018,
        2.220446049250313e-16,
        -3.3306690738754696e-16,
        -1.1102230246251565e-16
    ],
    "fnspace": {
        "dim_": 3,
        "b_": [
            1.9841629323767243,
            1.9903768915597309,
            0.9935673628631738
        ],
        "a_": [
            0.0043788616462432355,
            0.2533646009637481,
            0.006764061990002789
        ],
        "sa_": [
            -1.0,
            -1.0,
            -1.0
        ],
        "sb_": [
            1.0,
            1.0,
            1.0
        ],
        "pnames_": [
            "x",
            "y",
            "z"
        ]
    }
}"""
        return s

if __name__ == "__main__":
    unittest.main()