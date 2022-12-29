"""
The core algorithm
"""
import time
import numpy as np
from scipy.stats import gamma as Gamma
from scipy.stats import beta as Beta
from scipy.stats import norm
from scipy.special import loggamma
from statsmodels.stats.proportion import proportion_confint

from distribution import sample_l2_vec, sample_linfty_vec, sample_l1_vec
from utils import lambertWlog

# ============== fast beta section ==============

def get_beta(pA):
    return 0.08 * (-np.log(1 - pA) - 5) + 0.6 if pA >= 0.5 else 0.5


# experimental
def get_beta2(pA):
    return max(pA - 0.1, 0.4)

# translator from beta (p) here to concrete l2 ball norm
# mainly for debug and tuning use
def T(d, k, sigma, p):
    from scipy.stats import gamma
    import numpy as np
    return np.sqrt(d / (d - 2.0 * k)) * sigma * np.sqrt(2.0 * gamma(d / 2.0 - k).ppf(p))

FAST_BETA_TH = 0.0

# the fast beta threshold is useful for precise computation of general Gaussian and standard Gaussian mode
def calc_fast_beta_th(d, eps=1e-10):
    global FAST_BETA_TH
    l, r = 0.0, 0.5
    d = (d - 1.) / 2.
    while r - l > eps:
        mid = (l + r) / 2.
        if Beta(d, d).cdf(mid) < eps:
            l = mid
        else:
            r = mid
    FAST_BETA_TH = l
    print(f'Setting FAST_BETA_TH to {FAST_BETA_TH}')

def fast_beta(p, x):
    if p == 1:
        return max(min(x, 1.0), 0.0)
    elif p <= 300 or FAST_BETA_TH <= x <= 1.0 - FAST_BETA_TH:
        return Beta(p, p).cdf(x)
    elif x < FAST_BETA_TH:
        return 0.
    else:
        return 1.

# ============== fast beta section end ======================

def ln_exp_plus_exp(sgn1: int, pow1: float, sgn2: int, pow2: float, EPS=1e-6):
    # compute ln(s1 * exp(pow1) + s2 * exp(pow2))
    M = max(pow1, pow2)
    if abs(pow1 - pow2) > 20.:
        # they differ too much (over e^10 times), so use the approximation ln(1+x) = x, whose precision is O(e^(-20))
        pow1, pow2 = pow1 - M, pow2 - M
        if abs(pow1) < EPS:
            # M = pow1 >> pow2
            return (M + sgn2 * np.exp(pow2)) if sgn1 == 1 else -np.inf
        else:
            # pow1 << pow2 = M
            return (M + sgn1 * np.exp(pow1)) if sgn2 == 1 else -np.inf
    else:
        # do precise computation
        res = sgn1 * np.exp(pow1 - M) + sgn2 * np.exp(pow2 - M)
        return (M + np.log(res)) if res > 0. else -np.inf

def sum_exp_greater_than_one(sgn1, pow1, sgn2, pow2):
    # return true if sgn1 * exp(pow1) + sgn2 * exp(pow2) > 1.
    M = max(pow1, pow2)
    if -20. <= M <= 20.:
        # direct computation since the order is small
        return (sgn1 * np.exp(pow1) + sgn2 * np.exp(pow2)) >= 1.
    else:
        now = sgn1 * np.exp(pow1 - M) + sgn2 * np.exp(pow2 - M)
        if now <= 0. or M <= -20.:
            # if now < 0, definitely < 0 < 1
            # Since now < 2, if M <= -20, actual num exp(M) * now < 1
            return False
        else:
            # actual num exp(M) * now > 1 <=> M + ln(now) > 0
            return np.log(now) + M >= 0.

def np_sum_exp_greater_than_one(sgn1, pow1, sgn2, pow2):
    # the numpy version, where sgn1, sgn2 are scalars in -1 or 1; and pow1, pow2 are 1-D np arrays in same length
    M = np.max(np.vstack([pow1, pow2]),axis=0)
    ans = np.zeros_like(pow1, np.bool)
    ans[M <= -20.] = False
    ans[(-20. <= M) * (M <= 20.)] = ((sgn1 * np.exp(pow1[(-20. <= M) * (M <= 20.)]) + sgn2 * np.exp(pow2[(-20. <= M) * (M <= 20.)])) >= 1)
    Mbuf = M[M >= 20.]
    buf = (sgn1 * np.exp(pow1[M >= 20.] - Mbuf) + sgn2 * np.exp(pow2[M >= 20.] - Mbuf))
    buf[buf > 0] = np.log(buf[buf > 0]) + Mbuf[buf > 0]
    ans[M >= 20.] = (buf > 0.)
    return ans

# =============== auxiliary function end ====================


# =============== integration functions =====================


def P_single_var_numerical(disttype, d, k, sigma, logK, r):
    """
        compute P = integral {p(x) / p(x-x0) >= logK} p(x)
        via numerical integration
    :param disttype: gaussian / general-gaussian, belows with *_numerical suffix are the same
    :param d:
    :param k:
    :param sigma:
    :param logK:
    :param r:
    :return:
    """
    if disttype == 'gaussian':
        return norm.cdf(r / (2.0 * sigma) - (sigma * logK) / r)
    elif disttype == 'general-gaussian':
        # general-gaussian

        def inv_gK(x):
            # return g^-1(g(x)/K)
            ans = sigma * np.sqrt(2.0 * k *
                                  lambertWlog(logK / k + x * x / (2.0 * sigma * sigma * k) +
                                              np.log(x * x / (2.0 * sigma * sigma * k)))
                                  )
            return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * x) + r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2) / (4.0 * sigma * np.sqrt(2.0 * x) * r)
            ),
            lb=0., ub=np.inf
        )
        return ans

    elif disttype == 'L2-Linfty-gaussian':
        # L2-Linfty-gaussian
        return norm.cdf(np.sqrt(d) * r / (2.0 * sigma) - (sigma * logK) / (np.sqrt(d) * r))
    elif disttype == 'L2-Linfty-general-gaussian':
        # L2-Linfty-general-gaussian
        raise Exception('reached code that is not actively maintained')
        # def inv_gK(x):
        #     # return g^-1(g(x)/K)
        #     ans = sigma * np.sqrt(2.0 * k *
        #                           lambertWlog(logK / k + x * x / (2.0 * sigma * sigma * k) +
        #                                       np.log(x * x / (2.0 * sigma * sigma * k)))
        #                           )
        #     return ans
        #
        # ans = Gamma(d / 2.0 - k).expect(
        #     lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
        #         ((sigma * np.sqrt(2.0 * x) + np.sqrt(d) * r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2) /
        #         (4.0 * sigma * np.sqrt(2.0 * x) * np.sqrt(d) * r)
        #     ),
        #     lb=0., ub=np.inf
        # )
        # return ans

    else:
        # general-gaussian-th
        # this part is the same as general-gaussian, but simplified
        ans = Gamma(d / 2.0 - k).expect(
            lambda t: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * t) + r) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(logK / k + t / k + np.log(t / k)))
                /
                (4.0 * sigma * np.sqrt(2.0 * t) * r)
            ),
            lb=0., ub=np.inf
        )
        return ans

def Q_single_var_numerical(disttype, d, k, sigma, beta, logK, r):
    """
        compute Q = integral {p(x) / p(x-x0) >= logK} q(x)
        via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param logK:
    :param r:
    :return:
    """
    if disttype == 'gaussian':
        return norm.cdf(r / (2.0 * beta) - (sigma * sigma * logK) / (beta * r))
    elif disttype == 'general-gaussian':
        # general-gaussian

        def inv_gK(x):
            # need to return g^-1(g(x)/K)
            ans = sigma * np.sqrt(2.0 * k *
                                  lambertWlog(logK / k + x * x / (2.0 * sigma * sigma * k) +
                                              np.log(x * x / (2.0 * sigma * sigma * k)))
                                  )
            return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((beta * np.sqrt(2.0 * x) + r) ** 2 - inv_gK(beta * np.sqrt(2.0 * x)) ** 2) / (
                        4.0 * beta * np.sqrt(2.0 * x) * r)
            ),
            lb=0., ub=np.inf
        )
        return ans
    elif disttype == 'general-gaussian-th':
        # general-gaussian-th
        # beta = T^2 / (2 sigma ** 2)

        nu = 1.0 / Gamma(d / 2.0 - k).cdf(beta)
        ans = nu * Gamma(d / 2.0 - k).expect(
            lambda t: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * t) + r) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(logK / k + t / k + np.log(t / k)))
                / (4.0 * sigma * np.sqrt(2.0 * t) * r),
            ),
            lb=0., ub=beta
        )
        return ans

def Pprime_single_var_numerical(disttype, d, k, sigma, beta, logK, r):
    """
        Compute P'= integral {q(x) / p(x-x0) >= logK} p(x)
        via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param logK:
    :param r:
    :return:
    """
    if disttype == 'gaussian':
        # standard gaussian
        def sqr_g(x):
            # need to return g^(-1) (h(sigma * sqrt(2x)) / logK)^2
            ans = 2.0 * sigma * sigma * (sigma * sigma / (beta * beta) * x + logK - d * np.log(beta / sigma))
            return ans
        ans = Gamma(d / 2.0).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((r + sigma * np.sqrt(2.0 * x))**2 - sqr_g(x)) /
                (4.0 * r * sigma * np.sqrt(2.0 * x))
            ),
            lb=0., ub=np.inf
        )
        return ans
    elif disttype == 'general-gaussian':
        # general-gaussian
        def inv_gK_sq(x):
            # need to return g^(-1) (gbeta(sigma*(sqrt(2x))/logK)^2
            ans = (sigma ** 2) * 2.0 * k * lambertWlog(
                logK / k + (sigma * sigma * x / (beta * beta * k)) + np.log(sigma * sigma * x / (beta * beta * k)) + (d/k) * np.log(beta / sigma)
            )
            return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((r + sigma * np.sqrt(2.0 * x))**2 - inv_gK_sq(x)) /
                (4.0 * r * sigma * np.sqrt(2.0 * x))
            ),
            lb=0., ub=np.inf
        )
        return ans

    elif disttype == 'general-gaussian-th':
        # general-gaussian-th
        nu = 1.0 / Gamma(d / 2.0 - k).cdf(beta)
        nupow = -np.log(nu) / k
        ans = Gamma(d / 2.0 - k).expect(
            lambda t: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * t) + r) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(logK / k + t / k + np.log(t / k) + nupow))
                / (4.0 * sigma * np.sqrt(2.0 * t) * r),
                ),
            lb=0., ub=beta
        )
        return ans


def Qprime_single_var_numerical(disttype, d, k, sigma, beta, logK, r):
    """
        Compute Q'= integral {q(x) / p(x-x0) >= logK} q(x)
        via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param logK:
    :param r:
    :return:
    """
    if disttype == 'gaussian':
        # standard gaussian
        def sqr_g(x):
            # need to return g^(-1) (h(beta * sqrt(2x)) / logK)^2
            ans = 2.0 * sigma * sigma * (x + logK - d * np.log(sigma / beta))
            return ans
        ans = Gamma(d / 2.0).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((r + beta * np.sqrt(2.0 * x))**2 - sqr_g(x)) /
                (4.0 * r * beta * np.sqrt(2.0 * x))
            ),
            lb=0, ub=np.inf
        )
        return ans
    elif disttype == 'general-gaussian':
        # general-gaussian
        def inv_gK_sq(x):
            # need to return g^-1(gbeta(beta * sqrt(2x))/K)**2
            ans = (sigma ** 2) * 2.0 * k * lambertWlog(
                logK / k + (x / k) + np.log(x / k) + (d/k) * np.log(beta / sigma)
            )
            return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((beta * np.sqrt(2.0 * x) + r)**2 - inv_gK_sq(x)) /
                (4.0 * beta * np.sqrt(2.0 * x) * r)
            ),
            lb=0, ub=np.inf
        )
        return ans
    elif disttype == 'general-gaussian-th':
        # general-gaussian-th
        nu = 1.0 / Gamma(d / 2.0 - k).cdf(beta)
        nupow = -np.log(nu) / k
        ans = nu * Gamma(d / 2.0 - k).expect(
            lambda t: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * t) + r) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(logK / k + t / k + np.log(t / k) + nupow))
                / (4.0 * sigma * np.sqrt(2.0 * t) * r),
                ),
            lb=0., ub=beta
        )
        return ans


def Pshift_single_var_numerical(disttype, d, k, sigma, logK, r, limit=50):
    """
        Compute Pshift (or R) = integral {p(x) / p(x-x0) >= logK} p(x-x0)
        via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param logK:
    :param r:
    :return:
    """
    if disttype == 'gaussian':
        # gaussian
        return norm.cdf(- r / (2.0 * sigma) - (sigma * logK) / r)
    elif disttype == 'general-gaussian':
        # general-gaussian
        def inv_gK(x):
            # need to return g^-1(g(x)K)
            ans = sigma * np.sqrt(2.0 * k *
                                  lambertWlog(- logK / k + x * x / (2.0 * sigma * sigma * k) + np.log(
                                      x * x / (2.0 * sigma * sigma * k)))
                                  )
            ans = ans.real
            return ans

        ans = 1.0 - Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * x) + r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2)
                / (4.0 * sigma * np.sqrt(2.0 * x) * r)
            ),
            lb=0., ub=np.inf,
            limit=limit
        )
        return ans
    elif disttype == 'L2-Linfty-gaussian':
        # L2-Linfty-gaussian
        return norm.cdf(- np.sqrt(d) * r / (2.0 * sigma) - (sigma * logK) / (np.sqrt(d) * r))
    elif disttype == 'L2-Linfty-general-gaussian':
        # L2-Linfty-general-gaussian
        def inv_gK(x):
            # need to return g^-1(g(x)K)
            ans = sigma * np.sqrt(2.0 * k *
                                  lambertWlog(- logK / k + x * x / (2.0 * sigma * sigma * k) + np.log(
                                      x * x / (2.0 * sigma * sigma * k)))
                                  )
            ans = ans.real
            return ans

        ans = 1.0 - Gamma(d / 2.0 - k).expect(
            lambda x: Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                ((sigma * np.sqrt(2.0 * x) + np.sqrt(d) * r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2)
                / (4.0 * sigma * np.sqrt(2.0 * x) * np.sqrt(d) * r)
            ),
            lb=0., ub=np.inf,
            limit=limit
        )
        return ans
    elif disttype == 'general-gaussian-th':
        # general-gaussian-th

        ans = 1.0 - Gamma(d / 2.0 - k).expect(
            lambda t:
            Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(((sigma * np.sqrt(2.0 * t) + r)**2 - 2.0 * k * sigma * sigma * lambertWlog(t / k - logK / k + np.log(t/k)))
              / (4.0 * sigma * np.sqrt(2.0 * t) * r)),
            lb=0., ub=np.inf,
            limit=limit
        )
        return ans

def P_double_var_numerical(disttype, d, k, sigma, beta,
                           lamb1, lamb2, r, bisearch_precision):
    """
        Compute P(lamb1, lamb2) via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param lamb1:
    :param lamb2:
    :param r:
    :param bisearch_precision:
    :return:
    """

    s1 = 1 if lamb1 > 0. else -1
    l1 = abs(lamb1) - bisearch_precision
    s2 = 1 if lamb2 > 0. else -1
    l2 = abs(lamb2) - bisearch_precision
    K1 = sigma**2 / beta**2

    # print(f'      {s1} {l1} {s2} {l2} {K1}')

    if disttype == 'gaussian':
        # gaussian
        def g_inv_sq_with_sigma(x: float):
            inner_ln = ln_exp_plus_exp(s1, l1 - x,
                                       s2, l2 + d * (np.log(sigma) - np.log(beta)) - x * K1)
            if inner_ln == -np.inf:
                return np.inf
            else:
                ans = -2.0 * sigma * sigma * inner_ln
                return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x:
            fast_beta((d - 1.0) / 2.0,
                      ((r + sigma * np.sqrt(2.0 * x)) ** 2 - g_inv_sq_with_sigma(x))
                      / (4.0 * r * sigma * np.sqrt(2.0 * x))
                      )
            ,
            lb=0., ub=np.inf,
            limit=50
        )
        return ans
    else:
        # general-gaussian
        C1 = (d - 2.0 * k) * (np.log(sigma) - np.log(beta)) + l2

        def g_inv_sq_with_sigma(x: float):
            inner_ln = ln_exp_plus_exp(s1, l1 - x,
                                       s2, C1 - x * K1)
            if inner_ln == -np.inf:
                return np.inf
            else:
                hlog_inv = lambertWlog(np.log(x) - np.log(k) - inner_ln / k)
                ans = 2.0 * k * sigma * sigma * hlog_inv
                return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x:
            fast_beta((d - 1.0) / 2.0,
                      ((r + sigma * np.sqrt(2.0 * x)) ** 2 - g_inv_sq_with_sigma(x))
                      / (4.0 * r * sigma * np.sqrt(2.0 * x))
                      )
            ,
            lb=0., ub=np.inf,
            limit=50
        )
        return ans

def Q_double_var_numerical(disttype, d, k, sigma, beta,
                           lamb1, lamb2, r, bisearch_precision, limit=50):
    """
        Compute Q(lamb1, lamb2) via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param lamb1:
    :param lamb2:
    :param r:
    :param bisearch_precision:
    :return:
    """

    s1 = 1 if lamb1 > 0. else -1
    l1 = abs(lamb1) - bisearch_precision
    s2 = 1 if lamb2 > 0. else -1
    l2 = abs(lamb2) - bisearch_precision
    K2 = beta**2 / sigma**2

    if disttype == 'gaussian':
        # gaussian
        def g_inv_sq_with_sigma(x: float):
            inner_ln = ln_exp_plus_exp(s1, l1 - x * K2,
                                       s2, l2 + d * (np.log(sigma) - np.log(beta)) - x)
            if inner_ln == -np.inf:
                return np.inf
            else:
                ans = -2.0 * sigma * sigma * inner_ln
                return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x:
            fast_beta((d - 1.0) / 2.0,
                      ((r + beta * np.sqrt(2.0 * x)) ** 2 - g_inv_sq_with_sigma(x))
                      / (4.0 * r * beta * np.sqrt(2.0 * x))
                      )
            ,
            lb=0., ub=np.inf,
            limit=limit
        )
        return ans
    else:
        # general-gaussian
        C1 = (d - 2.0 * k) * (np.log(sigma) - np.log(beta)) + l2
        def g_inv_sq_with_beta(x: float):
            inner_ln = ln_exp_plus_exp(s1, l1 - x * K2,
                                       s2, C1 - x)
            if inner_ln == -np.inf:
                return np.inf
            else:
                hlog_inv = lambertWlog(np.log(x) - np.log(k) + 2.0 * np.log(beta) - 2.0 * np.log(sigma) - inner_ln / k)
                ans = 2.0 * k * sigma * sigma * hlog_inv
                return ans

        ans = Gamma(d / 2.0 - k).expect(
            lambda x:
            fast_beta((d - 1.0) / 2.0,
                      ((r + beta * np.sqrt(2.0 * x)) ** 2 - g_inv_sq_with_beta(x))
                      / (4.0 * r * beta * np.sqrt(2.0 * x))
                      )
            # Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
            #     ((r + beta * np.sqrt(2.0 * x)) ** 2 - g_inv_sq_with_beta(x))
            #     / (4.0 * r * beta * np.sqrt(2.0 * x))
            # )
            ,
            lb=0., ub=np.inf,
            limit=50
        )
        return ans

def Pshift_double_var_numerical(disttype, d, k, sigma, beta,
                                lamb1, lamb2, rad, bisearch_precision,
                                eps=1e-8):
    """
        Compute R(lamb1, lamb2) via numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param lamb1:
    :param lamb2:
    :param r:
    :param bisearch_precision:
    :param eps:
    :return:
    """
    s1 = 1 if lamb1 > 0. else -1
    l1 = abs(lamb1) - bisearch_precision
    s2 = 1 if lamb2 > 0. else -1
    l2 = abs(lamb2) - bisearch_precision

    if disttype == 'gaussian':
        # gaussian
        def m_inv_search_sq(x, s1, l1, s2, l2, d, sigma, beta, eps, bisearch_ub=1e+6):
            # binary search to get m_inv
            if ln_exp_plus_exp(s1, l1 + x, s2, l2 + x + d * np.log(sigma / beta)) < 0:
                # it means that m_inv does not include point 0
                # We filter this case out currently
                return - np.inf
            else:
                l, r = 0., bisearch_ub
                while r-l > eps:
                    mid = (l + r) / 2.0
                    if sum_exp_greater_than_one(
                        s1,
                        x - (mid ** 2 / (2.0 * sigma * sigma)) + l1,
                        s2,
                        x - (mid ** 2 / (2.0 * beta * beta)) + l2 + d * np.log(sigma / beta)):
                        l = mid
                    else:
                        r = mid
                ans = (l + r) / 2.0
                assert r < bisearch_ub
                # Otherwise it means that the binary search range is too narrow
                # for floating point soundness
                return l ** 2


        # ans = Gamma(d / 2.0).expect(
        #     lambda x:
        #     Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
        #         0.5 +
        #         (m_inv_search_sq(x, s1, l1, s2, l2, d, sigma, beta, eps) - rad ** 2 - 2.0 * x * sigma * sigma)
        #         / (4.0 * rad * sigma * np.sqrt(2.0 * x))
        #     )
        #     ,
        #     lb=0., ub=np.inf,
        #     limit=200
        # )

        ans = 1. - Gamma(d / 2.0).expect(
            lambda x:
            Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                0.5 +
                (- m_inv_search_sq(x, s1, l1, s2, l2, d, sigma, beta, eps) + rad ** 2 + 2.0 * x * sigma * sigma)
                / (4.0 * rad * sigma * np.sqrt(2.0 * x))
            )
            ,
            lb=0., ub=np.inf,
            limit=200
        )

    else:
        # general-gaussian

        def m_inv_search_sq(x, s1, l1, s2, l2, d, k, sigma, beta, eps, bisearch_ub=1e+6):
            # binary search to get m_inv
            # empirically, we set the binary search upper bound be bisearch_ub
            # # currently we only handle the case where lamb1 > 0, which we found holds for all samples
            # assert s1 == 1
            # fix: now directly use this for lamb1 < 0 case, since we find some such cases.
            # for these cases, the result might be a bit conservative
            if ln_exp_plus_exp(s1, l1 - (d - 2.0 * k) * np.log(sigma), s2, l2 - (d - 2.0 * k) * np.log(beta)) == -np.inf:
                # it means that lamb1 / (sigma)**(d-2k) + lamb2 / (beta)**(d-2k) < 0.
                # Therefore, the function < 0 at the origin (otherwise the function = +infty at the origin).
                # We filter this case out currently
                # # Therefore, the function < 0 anywhere, and there is no point available for integration.
                return - np.inf
            else:
                l, r = 0., bisearch_ub
                while r-l > eps:
                    mid = (l + r) / 2.0
                    if sum_exp_greater_than_one(
                            s1,
                            x + k * np.log(2.0 * x) + d * np.log(sigma) + l1
                            - (d - 2.0 * k) * np.log(sigma) - 2.0 * k * np.log(mid) - (mid ** 2 / (2.0 * sigma * sigma)),
                            s2,
                            x + k * np.log(2.0 * x) + d * np.log(sigma) + l2
                            - (d - 2.0 * k) * np.log(beta) - 2.0 * k * np.log(mid) - (mid ** 2 / (2.0 * beta * beta))):
                        l = mid
                    else:
                        r = mid
                ans = (l + r) / 2.0
                # print('r =', r)
                # print('M(r) =', s1 * np.exp(x + k * np.log(2.0 * x) + d * np.log(sigma) + l1 - (d - 2.0 * k) * np.log(sigma) - 2.0 * k * np.log(r) - (r ** 2) / (2.0 * sigma * sigma)) +
                #       s2 * np.exp(x + k * np.log(2.0 * x) + d * np.log(sigma) + l2 - (d - 2.0 * k) * np.log(beta) - 2.0 * k * np.log(r) - (r ** 2) / (2.0 * beta * beta)))
                assert r < bisearch_ub
                # Otherwise it means that the binary search range is too narrow
                # for floating point soundness
                return l ** 2

        ans = Gamma(d / 2.0 - k).expect(
            lambda x:
            Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                0.5 +
                (m_inv_search_sq(x, s1, l1, s2, l2, d, k, sigma, beta, eps) - rad ** 2 - 2.0 * x * sigma * sigma)
                / (4.0 * rad * sigma * np.sqrt(2.0 * x))
            )
            ,
            lb=0., ub=np.inf,
            limit=200
        )

    return ans

# =============== MC sampling functions =====================

# def confint(cnt, N, alpha=0.001):
#     ret = proportion_confint(int(cnt), N, alpha, method='beta')
#     if 0 < cnt < N:
#         return ret
#     elif cnt == 0:
#         return (0., ret[1])
#     else:
#         return (ret[0], 1.)
#
# def sampler_MC(disttype, d, k, sigma, beta, r, num, L2=True, Linf=True, batch_size=1000):
#
#         # ret := (L2samples, Linfsamples)
#         # L2samples := (p(x), q(x), p(x - x0), p(x + x0), q(x + x0)) all in logarithm scale
#         # Linftysamples := (p(x), q(x), p(x - x0)) all in logarithm scale
#
#     if disttype == 'gaussian':
#         def calc_p(arr):
#             ans = np.linalg.norm(arr, ord=2, axis=1)
#             ans = - (ans ** 2) / (2.0 * sigma * sigma) - d / 2.0 * np.log(2.0 * sigma * sigma * np.pi)
#             return ans
#
#         def calc_q(arr):
#             ans = np.linalg.norm(arr, ord=2, axis=1)
#             ans = - (ans ** 2) / (2.0 * beta * beta) - d / 2.0 * np.log(2.0 * beta * beta * np.pi)
#             return ans
#
#     elif disttype == 'L2-Linfty-gaussian':
#         def calc_p(arr):
#             ans = np.linalg.norm(arr, ord=2, axis=1)
#             ans = - (ans ** 2) / (2.0 * sigma * sigma) - d / 2.0 * np.log(2.0 * sigma * sigma * np.pi)
#             return ans
#
#         def calc_q(arr):
#             ans = np.linalg.norm(arr, ord=np.inf, axis=1)
#             ans = - (ans ** 2) / (2.0 * beta * beta) - \
#                   (d / 2.0 * np.log(2.0 * beta * beta) + (d - 1) * np.log(2) + np.log(d) + loggamma(d / 2.0))
#             return ans
#
#     elif disttype == 'L2-Linfty-general-gaussian':
#         def calc_p(arr):
#             ans = np.linalg.norm(arr, ord=2, axis=1)
#             ans = - (ans ** 2) / (2.0 * sigma * sigma) - 2.0 * k * np.log(ans) \
#                   + loggamma(d / 2.0) - loggamma(d / 2.0 - k) \
#                   - ((d / 2.0 - k) * np.log(2.0 * sigma * sigma) + d / 2.0 * np.log(np.pi))
#             return ans
#
#         def calc_q(arr):
#             ans = np.linalg.norm(arr, ord=np.inf, axis=1)
#             ans = - (ans ** 2) / (2.0 * beta * beta) - 2.0 * k * np.log(ans) \
#                   - ((d / 2.0 - k) * np.log(2.0 * beta * beta) + (d - 1) * np.log(2) + np.log(d) + loggamma(d / 2.0 - k))
#             return ans
#
#     elif disttype == 'infty-general-gaussian':
#         def calc_p(arr):
#             ans = np.linalg.norm(arr, ord=np.inf, axis=1)
#             ans = - (ans ** 2) / (2.0 * sigma * sigma) - 2.0 * k * np.log(ans) \
#                   - ((d / 2.0 - k) * np.log(2.0 * sigma * sigma) + (d - 1) * np.log(2) + np.log(d) + loggamma(d / 2.0 - k))
#             return ans
#
#         def calc_q(arr):
#             ans = np.linalg.norm(arr, ord=np.inf, axis=1)
#             ans = - (ans ** 2) / (2.0 * beta * beta) - 2.0 * k * np.log(ans) \
#                   - ((d / 2.0 - k) * np.log(2.0 * beta * beta) + (d - 1) * np.log(2) + np.log(d) + loggamma(d / 2.0 - k))
#             return ans
#
#     elif disttype == 'L1-general-gaussian':
#         def calc_p(arr):
#             ans = np.linalg.norm(arr, ord=1, axis=1)
#             ans = - (ans ** 2) / (2.0 * sigma * sigma) - 2.0 * k * np.log(ans) \
#                   - ((d / 2.0 - k) * np.log(2.0 * sigma * sigma) + np.log(np.sqrt(d) / 2)) + loggamma(d) - loggamma(d / 2.0 - k)
#             return ans
#
#         def calc_q(arr):
#             ans = np.linalg.norm(arr, ord=1, axis=1)
#             ans = - (ans ** 2) / (2.0 * beta * beta) - 2.0 * k * np.log(ans) \
#                   - ((d / 2.0 - k) * np.log(2.0 * beta * beta) + np.log(np.sqrt(d) / 2)) + loggamma(d) - loggamma(d / 2.0 - k)
#             return ans
#
#     else:
#         raise Exception('Unsupported disttype')
#
#     L2samples = None
#     Linfsamples = None
#
#     # a vital part to remember to change!
#     if disttype == 'gaussian':
#         x0 = np.zeros(d)
#         x0[0] += r
#     else:
#         x0 = np.ones(d) * r
#
#     print(f'  Sampler metainfo: {disttype} sigma={sigma}, beta={beta}, d={d}, k={k}')
#
#     if L2 is True:
#         # generate prime distribution samples
#         if disttype == 'L2-Linfty-gaussian' or disttype == 'gaussian':
#             dist = Gamma(d / 2.0)
#         else:
#             # L2-Linfty-general-gaussian
#             dist = Gamma(d / 2.0 - k)
#         now_n = 0
#         ps, qs, psn, psp, qsp = list(), list(), list(), list(), list()
#         while now_n < num:
#             now_batch_size = min(batch_size, num - now_n)
#             batch_rs = dist.rvs(now_batch_size)
#             batch_rs = sigma * np.sqrt(2) * np.sqrt(batch_rs)
#             if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' or disttype == 'gaussian':
#                 vec_samples = sample_l2_vec(d, now_batch_size)
#             elif disttype == 'infty-general-gaussian':
#                 vec_samples = sample_linfty_vec(d, now_batch_size)
#             elif disttype == 'L1-general-gaussian':
#                 vec_samples = sample_l1_vec(d, now_batch_size)
#             else:
#                 raise Exception('Unsupported disttype')
#             vec_samples = vec_samples * batch_rs[:, np.newaxis]
#             ps.append(calc_p(vec_samples))
#             qs.append(calc_q(vec_samples))
#             psn.append(calc_p(vec_samples - x0[np.newaxis, :]))
#             psp.append(calc_p(vec_samples + x0[np.newaxis, :]))
#             qsp.append(calc_q(vec_samples + x0[np.newaxis, :]))
#             now_n += now_batch_size
#             del vec_samples
#         L2samples = (np.concatenate(ps), np.concatenate(qs),
#                      np.concatenate(psn), np.concatenate(psp), np.concatenate(qsp))
#     if Linf is True:
#         # generate extra distribution samples
#         if disttype == 'L2-Linfty-gaussian' or disttype == 'gaussian':
#             dist = Gamma(d / 2.0)
#         else:
#             # L2-Linfty-general-gaussian
#             dist = Gamma(d / 2.0 - k)
#         now_n = 0
#         ps, qs, psn = list(), list(), list()
#         while now_n < num:
#             now_batch_size = min(batch_size, num - now_n)
#             batch_rs = dist.rvs(now_batch_size)
#             batch_rs = beta * np.sqrt(2) * np.sqrt(batch_rs)
#             if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian':
#                 vec_samples = sample_linfty_vec(d, now_batch_size)
#             elif disttype == 'infty-general-gaussian':
#                 vec_samples = sample_linfty_vec(d, now_batch_size)
#             elif disttype == 'L1-general-gaussian':
#                 vec_samples = sample_l1_vec(d, now_batch_size)
#             elif disttype == 'gaussian':
#                 vec_samples = sample_l2_vec(d, now_batch_size)
#             else:
#                 raise Exception('Unsupported disttype')
#             vec_samples = vec_samples * batch_rs[:, np.newaxis]
#             ps.append(calc_p(vec_samples))
#             qs.append(calc_q(vec_samples))
#             psn.append(calc_p(vec_samples - x0[np.newaxis, :]))
#             now_n += now_batch_size
#             del vec_samples
#         Linfsamples = (np.concatenate(ps), np.concatenate(qs), np.concatenate(psn))
#     return (L2samples, Linfsamples)
#
# def P_single_var_MC(disttype, samples, logK, alpha=0.01):
#     """
#         compute P = integral {p(x) / p(x-x0) >= logK} p(x)
#         via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param logK:
#     :param alpha:
#     :return:
#     """
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian':
#         ps = samples[0][0]
#         pshifts = samples[0][2]
#         cnts = np.sum(ps >= (pshifts + logK))
#         N = len(samples[0][0])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def Q_single_var_MC(disttype, samples, logK, alpha=0.01):
#     """
#         compute Q = integral {p(x) / p(x-x0) >= logK} q(x)
#         via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param logK:
#     :param alpha:
#     :return:
#     """
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian':
#         ps = samples[1][0]
#         pshifts = samples[1][2]
#         cnts = np.sum(ps >= (pshifts + logK))
#         N = len(samples[1][0])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
#
# def Pprime_single_var_MC(disttype, samples, logK, alpha=0.01):
#     """
#         Compute P'= integral {q(x) / p(x-x0) >= logK} p(x)
#         via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param logK:
#     :param alpha:
#     :return:
#     """
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian':
#         qs = samples[0][1]
#         pshifts = samples[0][2]
#         cnts = np.sum(qs >= (pshifts + logK))
#         N = len(samples[0][1])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def Qprime_single_var_MC(disttype, samples, logK, alpha=0.01):
#     """
#         Compute Q'= integral {q(x) / p(x-x0) >= logK} q(x)
#         via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param logK:
#     :param alpha:
#     :return:
#     """
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian':
#         qs = samples[1][1]
#         pshifts = samples[1][2]
#         cnts = np.sum(qs >= (pshifts + logK))
#         N = len(samples[1][1])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def Pshift_single_var_MC(disttype, samples, logK, alpha=0.001):
#     """
#         Compute Pshift (or R) = integral {p(x) / p(x-x0) >= logK} p(x-x0)
#         via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param logK:
#     :param alpha:
#     :return:
#     """
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian':
#         pshifts = samples[0][3]
#         ps = samples[0][0]
#         cnts = np.sum(pshifts >= (ps + logK))
#         N = len(samples[0][3])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def P_double_var_MC(disttype, samples, lamb1, lamb2,
#                     bisearch_precision, alpha=0.01):
#     """
#         Compute P(lamb1, lamb2) via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param lamb1:
#     :param lamb2:
#     :param bisearch_precision:
#     :param alpha:
#     :return:
#     """
#     s1 = 1 if lamb1 > 0. else -1
#     l1 = abs(lamb1) - bisearch_precision
#     s2 = 1 if lamb2 > 0. else -1
#     l2 = abs(lamb2) - bisearch_precision
#
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian' or disttype == 'gaussian':
#         pshifts = samples[0][2]
#         ps = samples[0][0]
#         qs = samples[0][1]
#         # print(f'lamb1={s1}e+({l1}) lamb2={s2}e+({l2})')
#         # print(ps, qs)
#         ps, qs = ps - pshifts, qs - pshifts
#         cnts = np.sum(np_sum_exp_greater_than_one(s1, ps + l1, s2, qs + l2))
#         N = len(samples[0][2])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def Q_double_var_MC(disttype, samples, lamb1, lamb2,
#                     bisearch_precision, alpha=0.01):
#     """
#         Compute Q(lamb1, lamb2) via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param lamb1:
#     :param lamb2:
#     :param bisearch_precision:
#     :param alpha:
#     :return:
#     """
#     s1 = 1 if lamb1 > 0. else -1
#     l1 = abs(lamb1) - bisearch_precision
#     s2 = 1 if lamb2 > 0. else -1
#     l2 = abs(lamb2) - bisearch_precision
#
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian' or disttype == 'gaussian':
#         pshifts = samples[1][2]
#         ps = samples[1][0]
#         qs = samples[1][1]
#         ps, qs = ps - pshifts, qs - pshifts
#         cnts = np.sum(np_sum_exp_greater_than_one(s1, ps + l1, s2, qs + l2))
#         N = len(samples[1][2])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')
#
# def Pshift_double_var_MC(disttype, samples, lamb1, lamb2,
#                          bisearch_precision, alpha=0.01):
#     """
#         Compute R(lamb1, lamb2) via Monte-Carlo with confidence interval
#     :param disttype:
#     :param samples:
#     :param lamb1:
#     :param lamb2:
#     :param bisearch_precision:
#     :param alpha:
#     :return:
#     """
#     s1 = 1 if lamb1 > 0. else -1
#     l1 = abs(lamb1) - bisearch_precision
#     s2 = 1 if lamb2 > 0. else -1
#     l2 = abs(lamb2) - bisearch_precision
#
#     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian' \
#             or disttype == 'infty-general-gaussian' or disttype == 'L1-general-gaussian' or disttype == 'gaussian':
#         pshifts = samples[0][3]
#         qshifts = samples[0][4]
#         ps = samples[0][0]
#         pshifts, qshifts = pshifts - ps, qshifts - ps
#         cnts = np.sum(np_sum_exp_greater_than_one(s1, pshifts + l1, s2, qshifts + l2))
#         N = len(samples[0][3])
#         return confint(cnts, N, alpha)
#     else:
#         raise Exception('Unsupported disttype')

# =============== Necessary binary search functions =========

def bin_search_K_on_P_numerical(disttype, d, k, sigma, rad, P, range=20., EPS=1e-6):
    """
        binary search for the logK that achieves the desired P value
        when numerical integration is available
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param r:
    :param P:
    :return:
    """
    logK_lb, logK_ub = -range, range
    while logK_ub - logK_lb > EPS:
        logK_mid = (logK_lb + logK_ub) / 2.0
        now_mass = P_single_var_numerical(disttype, d, k, sigma, logK_mid, rad)
        if now_mass < P:
            logK_ub = logK_mid
        else:
            logK_lb = logK_mid
    # conservatism
    ans = logK_ub
    return ans

def bin_search_K_on_P_MC(disttype, samples, P, alpha=0.01, range=20., EPS=1e-6, mode='o+'):
    """
        binary search for the logK that achieves the desired P value
        when only Monte-Carlo is available
    :param disttype:
    :param samples:
    :param P:
    :param alpha:
    :param range:
    :param EPS:
    :param mode:
    :return:
    """
    assert mode in ['--', '-+', 'o-', 'o+', '+-', '++']
    logK_lb, logK_ub = -range, range
    while logK_ub - logK_lb > EPS:
        logK_mid = (logK_lb + logK_ub) / 2.0
        now_mass_l, now_mass_r = P_single_var_MC(disttype, samples, logK_mid, alpha)
        if mode[0] == 'o':
            now_mass_mid = (now_mass_l + now_mass_r) / 2.0
            if P > now_mass_mid:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
        elif mode[0] == '-':
            if P > now_mass_l:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
        else:
            # mode[0] == '+'
            if P > now_mass_r:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
    return logK_lb if mode[1] == '-' else logK_ub


def bin_search_K_on_Qprime_numerical(disttype, d, k, sigma, beta, rad, Q, range=20., EPS=1e-6):
    """
        binary search for the logK that acheives the desired Qprime value
        when numerical integration is available
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param r:
    :param Q:
    :return:
    """
    logK_lb, logK_ub = -range, range
    while logK_ub - logK_lb > EPS:
        logK_mid = (logK_lb + logK_ub) / 2.0
        now_mass = Qprime_single_var_numerical(disttype, d, k, sigma, beta, logK_mid, rad)
        if now_mass < Q:
            logK_ub = logK_mid
        else:
            logK_lb = logK_mid
    # conservatism
    ans = logK_ub
    return ans

def bin_search_K_on_Qprime_MC(disttype, samples, Q, alpha=0.01, range=20., EPS=1e-6, mode='o+'):
    """
        binary search for the logK that achieves the desired Qprime value
        when only Monte-Carlo is available
    :param disttype:
    :param samples:
    :param Q:
    :param alpha:
    :param range:
    :param EPS:
    :param mode:
    :return:
    """
    assert mode in ['--', '-+', 'o-', 'o+', '+-', '++']
    logK_lb, logK_ub = -range, range
    while logK_ub - logK_lb > EPS:
        logK_mid = (logK_lb + logK_ub) / 2.0
        now_mass_l, now_mass_r = Qprime_single_var_MC(disttype, samples, logK_mid, alpha)
        if mode[0] == 'o':
            now_mass_mid = (now_mass_l + now_mass_r) / 2.0
            if Q > now_mass_mid:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
        elif mode[0] == '-':
            if Q > now_mass_l:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
        else:
            # mode[0] == '+'
            if Q > now_mass_r:
                logK_ub = logK_mid
            else:
                logK_lb = logK_mid
    return logK_lb if mode[1] == '-' else logK_ub

def bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1, rad, P,
                                       bisearch_precision, bisearch_boundary, eps,
                                       L=None, R=None):
    """
        Given the lambda 1 and desired P value, find the corresponding lambda 2
        that corresponds to the P(lambda 1, lambda 2) via binary search
        using numerical integration
    :param disttype:
    :param d:
    :param k:
    :param sigma:
    :param beta:
    :param lamb1_mid:
    :param r:
    :param p1:
    :param bisearch_precision:
    :param bisearch_boundary:
    :param eps:
    :param L:
    :param R:
    :return:
    """
    if L is not None:
        l, r = L, R
    else:
        l, r = - (bisearch_boundary + bisearch_precision), + (bisearch_boundary + bisearch_precision)
    while r - l > eps:
        mid = (l + r) / 2.0
        p_mid = P_double_var_numerical(disttype, d, k, sigma, beta, lamb1, mid, rad, bisearch_precision)
        # print('    ', f'[{l},{r}]', p_mid)
        if p_mid < P:
            l = mid
        else:
            r = mid
    return (l + r) / 2.0

def bin_search_lambda_2_on_P_MC(disttype, samples, lamb1, P,
                                bisearch_precision, bisearch_boundary, eps,
                                L=None, R=None, alpha=0.01, mode='o+'):
    """
        Given the lambda 1 and desired P value, find the corresponding lambda 2
        that corresponds to the P(lambda 1, lambda 2) via binary search
        when only Monte-Carlo is available
    :param disttype:
    :param samples:
    :param lamb1:
    :param P:
    :param bisearch_precision:
    :param bisearch_boundary:
    :param eps:
    :param L:
    :param R:
    :param alpha:
    :return:
    """
    assert mode in ['--', '-+', 'o-', 'o+', '+-', '++']
    if L is not None:
        l, r = L, R
    else:
        l, r = - (bisearch_boundary + bisearch_precision), + (bisearch_boundary + bisearch_precision)
    while r - l > eps:
        mid = (l + r) / 2.0
        p_mid_l, p_mid_r = P_double_var_MC(disttype, samples, lamb1, mid, bisearch_precision, alpha)
        if mode[0] == 'o':
            p_mid_mid = (p_mid_l + p_mid_r) / 2.0
            if p_mid_mid < P:
                l = mid
            else:
                r = mid
        elif mode[0] == '-':
            if p_mid_l < P:
                l = mid
            else:
                r = mid
        else:
            if p_mid_r < P:
                l = mid
            else:
                r = mid
    return l if mode[1] == '-' else r

# =============== Core binary search function ===============

"""
    Core double binary search based algorithm
"""
def check(disttype: str, r: float, d: int, k: int, sigma: float, beta: float,
          p1L: float, p1R: float, q1L: float, q1R: float,
          bisearch_precision=10., bisearch_boundary=5000., range_width=32.0, eps=1e-6):
    method = {'gaussian': 'numerical',
              'general-gaussian': 'numerical',
              'L2-Linfty-gaussian': 'montecarlo',
              'L2-Linfty-general-gaussian': 'montecarlo',
              'infty-gaussian': 'montecarlo',
              'infty-general-gaussian': 'montecarlo',
              'L1-general-gaussian': 'montecarlo',
              'general-gaussian-th': 'numerical',
              'gaussian-th': 'numerical'}[disttype]
    # method = {'gaussian': 'numerical', 'general-gaussian': 'montecarlo', 'L2-Linfty-gaussian': 'montecarlo'}[disttype]

    serious_alpha = 0.001
    serious_N = 100000

    if disttype.endswith('-th'):
        ##### heuristic region #####
        ##### should be made consistent with the sampler function sampler.py #####
        if beta == 'x' or (beta == 'x+' and p1R < 1. - 1e-8):
            beta = get_beta(p1L)
            print('grabbed beta:', beta)
        if beta == 'x2' or (beta == 'x2+' and p1R < 1. - 1e-8):
            beta = get_beta2(p1L)
            print('grabbed beta:', beta)
        if isinstance(beta, float):
            # beta translation
            beta = Gamma(d / 2.0 - k).ppf(beta)
            # print(f'translated beta: {beta}')

    "STEP 1: determine which p1 and q1 in the interval to use"
    naive = False
    p1shift = 0.
    if method == 'numerical':
        print(f"  P [{p1L}, {p1R}] Q [{q1L}, {q1R}]")

        if isinstance(beta, str) and beta.endswith('+'):
            # encounter the refined sampling case
            p1L, p1R = q1L, q1R
            print(f"    Original double sampling case P [{p1L}, {p1R}]")
            logK1 = bin_search_K_on_P_numerical(disttype, d, k, sigma, r, p1L)
            p1shift = Pshift_single_var_numerical(disttype, d, k, sigma, logK1, r, limit=200)
            naive = True
        else:
            logK1 = bin_search_K_on_P_numerical(disttype, d, k, sigma, r, p1L)
            q1ideal = Q_single_var_numerical(disttype, d, k, sigma, beta, logK1, r)
            print(f"  Q1 ideal={q1ideal}")
            # print(f"  logK1 = {logK1}")

            if q1ideal >= q1R:
                p1 = p1L
                q1 = q1R
            elif q1ideal >= q1L:
                p1shift = Pshift_single_var_numerical(disttype, d, k, sigma, logK1, r, limit=200)
                naive = True
                print(f"  Naive case A: {q1ideal} in [{q1L}, {q1R}]")
            else:
                # q1ideal < q1L < q1R
                logK2 = bin_search_K_on_Qprime_numerical(disttype, d, k, sigma, beta, r, q1L)
                p1ideal = Pprime_single_var_numerical(disttype, d, k, sigma, beta, logK2, r)
                if p1ideal < p1L:
                    p1 = p1L
                    q1 = q1L
                elif p1ideal > p1R:
                    p1 = p1R
                    q1 = q1L
                else:
                    logK3 = bin_search_K_on_P_numerical(disttype, d, k, sigma, r, p1ideal, EPS=eps*1e-3)
                    print(f"  logK3 = {logK3}")
                    p1shift = Pshift_single_var_numerical(disttype, d, k, sigma, logK3, r, limit=200)
                    naive = True
                    print(f"  Naive case B: {p1ideal} in [{p1L}, {p1R}]")
    else:
        raise Exception('Monte-Carlo based methods can support non-L2 certification. '
                        'But the code is currently not maintained. This implementation is just for reference. '
                        'No correctness gaurantee.')
        # print("generate samples...")
        # samples = sampler_MC(disttype, d, k, sigma, beta, r, serious_N, L2=True, Linf=True)
        # print("generate samples finish")
        #
        # logK1L, logK1U = bin_search_K_on_P_MC(disttype, samples, p1L, mode='--'), \
        #                  bin_search_K_on_P_MC(disttype, samples, p1L, mode='++')
        # # print(logK1L, logK1U)
        # q1idealL, _ = Q_single_var_MC(disttype, samples, logK1U)
        # _, q1idealR = Q_single_var_MC(disttype, samples, logK1L)
        # print(f"  Q1 ideal=[{q1idealL}, {q1idealR}]")
        #
        # if q1idealL >= q1R:
        #     p1 = p1L
        #     q1 = q1R
        # elif q1idealR >= q1L:
        #     if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian':
        #         p1shift = Pshift_single_var_numerical(disttype, d, k, sigma, logK1U, r, limit=200)
        #     else:
        #         print("generate samples 2nd...")
        #         samples = sampler_MC(disttype, d, k, sigma, beta, r, serious_N, L2=True, Linf=False)
        #         print("generate samples 2nd finish")
        #         p1shift, _ = Pshift_single_var_MC(disttype, samples, logK1U, alpha=0.001)
        #     naive = True
        #     print(f"  Naive case A: [{q1idealL}, {q1idealR}] overlaps [{q1L}, {q1R}]")
        # else:
        #     # q1ideal < q1L < q1R
        #     logK2L, logK2U = bin_search_K_on_Qprime_MC(disttype, samples, q1L, mode='--'), \
        #                      bin_search_K_on_Qprime_MC(disttype, samples, q1L, mode='++')
        #     p1idealL, _ = Pprime_single_var_MC(disttype, samples, logK2U)
        #     _, p1idealR = Pprime_single_var_MC(disttype, samples, logK2L)
        #     if p1idealR < p1L:
        #         p1 = p1L
        #         q1 = q1L
        #     elif p1idealL > p1R:
        #         p1 = p1R
        #         q1 = q1L
        #     else:
        #         logK3U = bin_search_K_on_P_MC(disttype, samples, p1idealL, mode='++')
        #         if disttype == 'L2-Linfty-gaussian' or disttype == 'L2-Linfty-general-gaussian':
        #             p1shift = Pshift_single_var_numerical(disttype, d, k, sigma, logK3U, r, limit=200)
        #         else:
        #             print("generate samples 2nd...")
        #             samples = sampler_MC(disttype, d, k, sigma, beta, r, serious_N, L2=True, Linf=False)
        #             print("generate samples 2nd finish")
        #             p1shift, _ = Pshift_single_var_MC(disttype, samples, logK3U, alpha=0.001)
        #         naive = True
        #         print(f"  Naive Case B: [{p1idealL}, {p1idealR}] overlaps [{p1L}, {p1R}]")

    if naive:
        return p1shift - eps >= 0.5

    "STEP 2: compute with determined p1 and q1"
    # centered around e^0 = 1
    lambda_1_L = - range_width
    lambda_1_R = + range_width

    if disttype == 'general-gaussian-th':
        # customized computation method for thresholded general-gaussian
        # we may need TODO: support gaussian-th, if we would like to handle low input dimension case.

        nu = 1.0 / Gamma(d / 2.0 - k).cdf(beta)

        """ determine sum of lambda_1 + nu * lambda_2 """

        def compute_Q_by_lamb12(now_lamb12):
            return nu * Gamma(d / 2.0 - k).expect(lambda t:
                  Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                      ( (r + sigma * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(t / k - now_lamb12 / k + np.log(t / k)) ) /
                      (4.0 * sigma * r * np.sqrt(2.0 * t)))
                  ,lb=0., ub=beta)

        # only need to binary search over the positive branch
        lamb12_L = - range_width
        lamb12_R = + range_width
        while lamb12_R - lamb12_L > eps:
            lamb12_mid = (lamb12_L + lamb12_R) / 2.0

            now_Q = compute_Q_by_lamb12(lamb12_mid)
            # print(f'     targ Q = {q1}, now_Q = {now_Q} with log(lamb1 + nu * lamb2) = {lamb12_mid}')

            if now_Q > q1:
                lamb12_R = lamb12_mid
            else:
                lamb12_L = lamb12_mid

        """ enlarge lambda12 to tolerate numerical intergration error """
        lamb12_gap = (lamb12_R - lamb12_L) / 2.
        now_Ql = compute_Q_by_lamb12(lamb12_L)
        while now_Ql + eps > q1:
            lamb12_L -= lamb12_gap
            now_Ql = compute_Q_by_lamb12(lamb12_L)
        now_Qr = compute_Q_by_lamb12(lamb12_R)
        while now_Qr - eps < q1:
            lamb12_R += lamb12_gap
            now_Qr = compute_Q_by_lamb12(lamb12_R)

        """ decompose P1 to two parts P1it, P1ni """
        p1ni = q1 / nu
        p1it = p1 - p1ni

        print(f'    p1 = {p1} = {p1it} + {p1ni}')

        """ determine lambda_1 """
        if p1it < eps:
            lamb1 = - range_width * 99.
            raise Exception('# caution: rare case, lamb1 is zero or negative')
        else:
            lamb1_L = - range_width
            lamb1_R = + range_width
            while lamb1_R - lamb1_L > eps:
                lamb1_mid = (lamb1_L + lamb1_R) / 2.0

                now_P1it = Gamma(d / 2.0 - k).expect(lambda t:
                    Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                        ( (r + sigma * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(t / k - lamb1_mid / k + np.log(t / k)) ) /
                        (4.0 * sigma * r * np.sqrt(2.0 * t)))
                    ,lb=beta, ub=np.inf)

                # print(f'     targ P1it = {p1it}, now_P1it = {now_P1it} with log(lamb1) = {lamb1_mid}')

                if now_P1it - eps > p1it:
                    lamb1_R = lamb1_mid
                else:
                    lamb1_L = lamb1_mid

            lamb1 = (lamb1_L + lamb1_R) / 2.0

            # by Theorem 5, we only need lamb1_L and lamb12_L to compute R

        """ integrate R """

        def u1(t):
            return Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                (2.0 * sigma * sigma * min(beta, k * lambertWlog(t/k + lamb12_L/k + np.log(t/k))) - (sigma * np.sqrt(2.0 * t) - r) ** 2)
                /
                (4.0 * sigma * r * np.sqrt(2.0 * t))
            )

        def u2(t):
            if lamb1 < -range_width * 10.:
                # lamb1 = 0
                return 0.0
            else:
                return max(0.0, Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                    (2.0 * sigma * sigma * k * lambertWlog(t/k + lamb1_L/k + np.log(t/k)) - (sigma * np.sqrt(2.0 * t) - r) ** 2)
                    /
                    (4.0 * sigma * r * np.sqrt(2.0 * t))
                ) - Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                    (2.0 * sigma * sigma * beta - (sigma * np.sqrt(2.0 * t) - r) ** 2)
                    /
                    (4.0 * sigma * r * np.sqrt(2.0 * t))
                ))

        # R lower bounds: for soundness
        R = Gamma(d / 2.0 - k).expect(lambda t: u1(t) + u2(t), lb=0., ub=np.inf)
        print(f'    R = {R}')

        return R - eps >= 0.5


    if method == 'numerical':
        lambda_2_for_lambda_1_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_L, r, p1,
                                                                     bisearch_precision, bisearch_boundary, eps)
        q1_for_lambda1_L = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                  lambda_1_L, lambda_2_for_lambda_1_L, r,
                                                  bisearch_precision)
        lambda_2_for_lambda_1_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_R, r, p1,
                                                                     bisearch_precision, bisearch_boundary, eps)
        q1_for_lambda1_R = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                  lambda_1_R, lambda_2_for_lambda_1_R, r,
                                                  bisearch_precision)
    else:
        raise Exception('reached code that is not actively maintained.')
        # lambda_2_for_lambda_1_L = bin_search_lambda_2_on_P_MC(disttype, samples, lambda_1_L, p1,
        #                                                       bisearch_precision, bisearch_boundary, eps)
        # tmp_l, tmp_r = Q_double_var_MC(disttype, samples, lambda_1_L, lambda_2_for_lambda_1_L, bisearch_precision)
        # q1_for_lambda1_L = (tmp_l + tmp_r) / 2.0
        # lambda_2_for_lambda_1_R = bin_search_lambda_2_on_P_MC(disttype, samples, lambda_1_R, p1,
        #                                                       bisearch_precision, bisearch_boundary, eps)
        # tmp_l, tmp_r = Q_double_var_MC(disttype, samples, lambda_1_R, lambda_2_for_lambda_1_R, bisearch_precision)
        # q1_for_lambda1_R = (tmp_l + tmp_r) / 2.0

    if (q1_for_lambda1_L - q1) * (q1_for_lambda1_R - q1) >= 0:
        # on the same side, it means that the intersection may not exist
        err_msg = f"""
This resulting interval does not contain an intersection point between p''s curve and q''s curve!
@ lambda1 {lambda_1_L} find lambda2 = {lambda_2_for_lambda_1_L} q1 = {q1_for_lambda1_L}.
@ lambda1 {lambda_1_R} find lambda2 = {lambda_2_for_lambda_1_R} q1 = {q1_for_lambda1_R}.
Desired q1 = {q1}
            """
        print(err_msg)
        raise Exception(err_msg)
    else:

        print(f"""
On lamb1 left  bound {lambda_1_L}, find lambda2 = {lambda_2_for_lambda_1_L}, where q1 = {q1_for_lambda1_L}.
On lamb1 right bound {lambda_1_R}, find lambda2 = {lambda_2_for_lambda_1_R}, where q1 = {q1_for_lambda1_R}.
""")

        if q1_for_lambda1_L > q1:
            # CASE A: P's curve: left side higher, right side lower

            # search range of lambda2
            lamb1_L, lamb1_R = lambda_1_L, lambda_1_R
            lamb2_L, lamb2_R = lambda_2_for_lambda_1_R, lambda_2_for_lambda_1_L

            if method == 'numerical':
                # for numerical case, we just need one pass of binary search
                while lamb1_R - lamb1_L > eps:
                    lamb1_mid = (lamb1_L + lamb1_R) / 2.0
                    lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, r, p1,
                                                                   bisearch_precision, bisearch_boundary, eps,
                                                                   L=lamb2_L, R=lamb2_R)
                    q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, r,
                                                    bisearch_precision)
                    # print(f"    ({lamb1_mid}, {lamb2_mid}) Q = {q1_mid}")
                    if q1_mid > q1:
                        lamb1_L = lamb1_mid
                        lamb2_R = lamb2_mid
                    else:
                        lamb1_R = lamb1_mid
                        lamb2_L = lamb2_mid
            else:
                raise Exception('reached code that is not actively maintained.')

                # # for monte-carlo case, we need to take confidence interval into consideration
                # # So two passes are needed, one for lb, and one for ub
                # tmp_lamb1_L, tmp_lamb1_R, tmp_lamb2_L, tmp_lamb2_R = lamb1_L, lamb1_R, lamb2_L, lamb2_R
                #
                # # first pass
                # while tmp_lamb1_R - tmp_lamb1_L > eps:
                #     lamb1_mid = (tmp_lamb1_L + tmp_lamb1_R) / 2.0
                #     lamb2_mid = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_mid, p1,
                #                                             bisearch_precision, bisearch_boundary, eps,
                #                                             L=tmp_lamb2_L, R=tmp_lamb2_R, mode='+-')
                #     q1_mid_L, q1_mid_U = Q_double_var_MC(disttype, samples, lamb1_mid, lamb2_mid, bisearch_precision)
                #     if q1_mid_L > q1:
                #         tmp_lamb1_L = lamb1_mid
                #         tmp_lamb2_R = lamb2_mid
                #     else:
                #         tmp_lamb1_R = lamb1_mid
                #         tmp_lamb2_L = lamb2_mid
                #
                # ans_lamb1_L = tmp_lamb1_L
                #
                # # second pass
                # tmp_lamb1_L, tmp_lamb1_R, tmp_lamb2_L, tmp_lamb2_R = lamb1_L, lamb1_R, lamb2_L, lamb2_R
                # while tmp_lamb1_R - tmp_lamb1_L > eps:
                #     lamb1_mid = (tmp_lamb1_L + tmp_lamb1_R) / 2.0
                #     lamb2_mid = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_mid, p1,
                #                                             bisearch_precision, bisearch_boundary, eps,
                #                                             L=tmp_lamb2_L, R=tmp_lamb2_R, mode='-+')
                #     q1_mid_L, q1_mid_U = Q_double_var_MC(disttype, samples, lamb1_mid, lamb2_mid, bisearch_precision)
                #     if q1_mid_U > q1:
                #         tmp_lamb1_L = lamb1_mid
                #         tmp_lamb2_R = lamb2_mid
                #     else:
                #         tmp_lamb1_R = lamb1_mid
                #         tmp_lamb2_L = lamb2_mid
                # ans_lamb1_R = tmp_lamb1_R
                #
                # lamb1_L, lamb1_R = ans_lamb1_L, ans_lamb1_R

            # sanity check
            if method == 'numerical':
                lamb2_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_L, r, p1,
                                                             bisearch_precision, bisearch_boundary, eps * 1e-3)
                q1_L = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_L, lamb2_L, r, bisearch_precision, limit=200)
                lamb2_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_R, r, p1,
                                                             bisearch_precision, bisearch_boundary, eps * 1e-3)
                q1_R = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_R, lamb2_R, r, bisearch_precision, limit=200)
                if q1_L >= q1 - eps * 1e+2 and q1_R <= q1 + eps * 1e+2:
                    lamb1 = lamb1_L
                    lamb2 = lamb2_R
                else:
                    error_msg = f"""
After binary search, the sanity check failed. (case A)
The desired q1 = {q1}, but
for ({lamb1_L}, {lamb2_L}) the q1 = {q1_L} (expect: > desired q1);
for ({lamb1_R}, {lamb2_R}) the q1 = {q1_R} (expect: < desired q1).
"""
                    print(error_msg)
                    raise Exception(error_msg)
            else:
                raise Exception('reached code that is not actively maintained.')

#                 lamb2_L = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_L, p1,
#                                                       bisearch_precision, bisearch_boundary, eps)
#                 q1_L_L, q1_L_R = Q_double_var_MC(disttype, samples, lamb1_L, lamb2_L, bisearch_precision)
#                 lamb2_R = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_R, p1,
#                                                       bisearch_precision, bisearch_boundary, eps)
#                 q1_R_L, q1_R_R = Q_double_var_MC(disttype, samples, lamb1_R, lamb2_R, bisearch_precision)
#                 if q1_L_R >= q1 and q1_R_L <= q1:
#                     lamb1 = lamb1_L
#                     lamb2 = lamb2_R
#                     print(f"""
# After binary search, the desired q1 = {q1}
# Left  side: lamb1 = {lamb1_L}, lamb2 = {lamb2_L}, q1 = [{q1_L_L}, {q1_L_R}]
# Right side: lamb1 = {lamb1_R}, lamb2 = {lamb2_R}, q1 = [{q1_R_L}, {q1_R_R}]
# """)
#                     # use average to approach the real solution
#                     lamb1 = (lamb1_L + lamb1_R) / 2.0
#                     lamb2 = (lamb2_L + lamb2_R) / 2.0
#                 else:
#                     error_msg = f"""
# After binary search, the sanity check failed. (case A)
# The desired q1 = {q1}, but
# for ({lamb1_L}, {lamb2_L}) the q1: [{q1_L_L}, {q1_L_R}] (include q1);
# for ({lamb1_R}, {lamb2_R}) the q1: [{q1_R_L}, {q1_R_R}] (include q1).
# """
#                     print(error_msg)
#                     raise Exception(error_msg)
        else:
            # CASE B: P's curve: left side lower, right side higher

            # search range of lambda2
            lamb1_L, lamb1_R = lambda_1_L, lambda_1_R
            lamb2_L, lamb2_R = lambda_2_for_lambda_1_R, lambda_2_for_lambda_1_L

            if method == 'numerical':
                # for numerical case, we just need one pass of binary search
                while lamb1_R - lamb1_L > eps:
                    lamb1_mid = (lamb1_L + lamb1_R) / 2.0
                    lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, r, p1,
                                                                   bisearch_precision, bisearch_boundary, eps,
                                                                   L=lamb2_L, R=lamb2_R)
                    q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, r,
                                                    bisearch_precision)
                    print(f"    ({lamb1_mid}, {lamb2_mid}) Q = {q1_mid}")
                    if q1_mid < q1:
                        lamb1_L = lamb1_mid
                        lamb2_R = lamb2_mid
                    else:
                        lamb1_R = lamb1_mid
                        lamb2_L = lamb2_mid
            else:
                raise Exception('reached code that is not actively maintained.')

                # # for monte-carlo case, we need to take confidence interval into consideration
                # # So two passes are needed, one for lb, and one for ub
                # tmp_lamb1_L, tmp_lamb1_R, tmp_lamb2_L, tmp_lamb2_R = lamb1_L, lamb1_R, lamb2_L, lamb2_R
                #
                # # first pass
                # while tmp_lamb1_R - tmp_lamb1_L > eps:
                #     lamb1_mid = (tmp_lamb1_L + tmp_lamb1_R) / 2.0
                #     lamb2_mid = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_mid, p1,
                #                                             bisearch_precision, bisearch_boundary, eps,
                #                                             L=tmp_lamb2_L, R=tmp_lamb2_R, mode='-+')
                #     q1_mid_L, q1_mid_U = Q_double_var_MC(disttype, samples, lamb1_mid, lamb2_mid, bisearch_precision)
                #     if q1_mid_U < q1:
                #         tmp_lamb1_L = lamb1_mid
                #         tmp_lamb2_R = lamb2_mid
                #     else:
                #         tmp_lamb1_R = lamb1_mid
                #         tmp_lamb2_L = lamb2_mid
                #
                # ans_lamb1_L = tmp_lamb1_L
                #
                # # second pass
                # tmp_lamb1_L, tmp_lamb1_R, tmp_lamb2_L, tmp_lamb2_R = lamb1_L, lamb1_R, lamb2_L, lamb2_R
                # while tmp_lamb1_R - tmp_lamb1_L > eps:
                #     lamb1_mid = (tmp_lamb1_L + tmp_lamb1_R) / 2.0
                #     lamb2_mid = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_mid, p1,
                #                                             bisearch_precision, bisearch_boundary, eps,
                #                                             L=tmp_lamb2_L, R=tmp_lamb2_R, mode='+-')
                #     q1_mid_L, q1_mid_U = Q_double_var_MC(disttype, samples, lamb1_mid, lamb2_mid, bisearch_precision)
                #     if q1_mid_L < q1:
                #         tmp_lamb1_L = lamb1_mid
                #         tmp_lamb2_R = lamb2_mid
                #     else:
                #         tmp_lamb1_R = lamb1_mid
                #         tmp_lamb2_L = lamb2_mid
                # ans_lamb1_R = tmp_lamb1_R
                #
                # lamb1_L, lamb1_R = ans_lamb1_L, ans_lamb1_R

            # sanity check
            if method == 'numerical':
                lamb2_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_L, r, p1,
                                                             bisearch_precision, bisearch_boundary, eps * 1e-3)
                q1_L = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_L, lamb2_L, r, bisearch_precision, limit=200)
                lamb2_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_R, r, p1,
                                                             bisearch_precision, bisearch_boundary, eps * 1e-3)
                q1_R = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_R, lamb2_R, r, bisearch_precision, limit=200)
                if q1_L <= q1 + eps * 1e+2 and q1_R >= q1 - eps * 1e-2:
                    lamb1 = lamb1_L
                    lamb2 = lamb2_R
                    # use average to approach the real solution
                    lamb1 = (lamb1_L + lamb1_R) / 2.0
                    lamb2 = (lamb2_L + lamb2_R) / 2.0
                else:
                    error_msg = f"""
After binary search, the sanity check failed. (case B)
The desired q1 = {q1}, but
for ({lamb1_L}, {lamb2_L}) the q1 = {q1_L} (expect: < desired q1);
for ({lamb1_R}, {lamb2_R}) the q1 = {q1_R} (expect: > desired q1).
"""
                    print(error_msg)
                    raise Exception(error_msg)
            else:
                raise Exception('reached code that is not actively maintained.')

#                 lamb2_L = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_L, p1,
#                                                       bisearch_precision, bisearch_boundary, eps)
#                 q1_L_L, q1_L_R = Q_double_var_MC(disttype, samples, lamb1_L, lamb2_L, bisearch_precision)
#                 lamb2_R = bin_search_lambda_2_on_P_MC(disttype, samples, lamb1_R, p1,
#                                                       bisearch_precision, bisearch_boundary, eps)
#                 q1_R_L, q1_R_R = Q_double_var_MC(disttype, samples, lamb1_R, lamb2_R, bisearch_precision)
#                 if q1_L_R >= q1 and q1_R_L <= q1:
#                     lamb1 = lamb1_L
#                     lamb2 = lamb2_R
#                 else:
#                     error_msg = f"""
# After binary search, the sanity check failed. (case B)
# The desired q1 = {q1}, but
# for ({lamb1_L}, {lamb2_L}) the q1: [{q1_L_L}, {q1_L_R}] (expect: include q1);
# for ({lamb1_R}, {lamb2_R}) the q1: [{q1_R_L}, {q1_R_R}] (expect: include q1).
# """
#                     print(error_msg)
#                     raise Exception(error_msg)

        print(f'    determined:\n      lamb1 = {lamb1}\n      lamb2 = {lamb2}')

        if method == 'numerical':
            p1shift = Pshift_double_var_numerical(disttype, d, k, sigma, beta, lamb1, lamb2, r, bisearch_precision,
                                                  eps=1e-8)
        else:
            raise Exception('reached code that is not actively maintained.')

            # print("generate samples 2nd...")
            # samples = sampler_MC(disttype, d, k, sigma, beta, r, serious_N, L2=True, Linf=True)
            # print("generate samples 2nd finish")
            # p1shift, p1shift_U = Pshift_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision, alpha=0.001)
            # # a more ambitious setting
            # # p1shift = (p1shift + p1shift_U) / 2.0
        print(f'    p1shift = {p1shift}')
        return p1shift >= 0.5


if __name__ == '__main__':
    # # some simple tests

    # # print(confint(0.5, 1000))
    # stime = time.time()
    # ret = sampler_MC('L2-Linfty-gaussian', d=3072, k=None, sigma=0.5, beta=0.0156199162191594, r=0.1, num=10)
    # # print(time.time() - stime)
    # #0156199162191594
    # print(ret[0])
    # print(ret[1])
    #
    # sgn1 = -1
    # sgn2 = 1
    # a = np.random.uniform(-2000., 2000., size=(1000000,))
    # b = np.random.uniform(-2000., 2000., size=(1000000,))
    # ans_0 = np.zeros_like(a, dtype=np.bool)
    # stime = time.time()
    # for i in range(len(a)):
    #     ans_0[i] = sum_exp_greater_than_one(sgn1, a[i], sgn2, b[i])
    # print(time.time() - stime)
    # stime = time.time()
    # ans_1 = np_sum_exp_greater_than_one(sgn1, a, sgn2, b)
    # print(time.time() - stime)
    # # print(ans_0)
    # print((ans_0 == ans_1).all())

    disttype = 'L2-Linfty-general-gaussian'
    d = 3072
    k = 1530
    sigma = 8.0
    beta = 0.24991865950655037
    r = 0.002
    for ratio in np.linspace(1.0, 10.0, num=10):
        print(ratio)
        new_samps, _ = sampler_MC(disttype, d, k, sigma, beta * ratio, r, 5000, Linf=False)
        a, b = new_samps[0], new_samps[1]
        print(min(a), max(a), (a).mean())
        print(min(b), max(b), (b).mean())