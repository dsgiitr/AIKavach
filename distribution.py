# import os
# #must set these before loading numpy:
# os.environ["OMP_NUM_THREADS"] = '8' # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = '8' # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
# #os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
# #os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6

import numpy as np
from numpy import linalg
import torch
from torch.distributions import dirichlet
from scipy.stats import norm
from scipy.stats import gamma, beta
from statsmodels.stats.proportion import proportion_confint

from utils import lambertWlog

# -------------auxiliary functions----------

def confint(cnt, N, alpha=0.001):
    ret = proportion_confint(int(cnt), N, alpha, method='beta')
    if 0 < cnt < N:
        return ret
    elif cnt == 0:
        return (0., ret[1])
    else:
        return (ret[0], 1.)

# --------------internal functions--------------

def _compute_at_origin(d, k, sigma, logK, r):
    def inv_gK(x):
        # need to return g^-1(g(x)/K)
        ans = sigma * np.sqrt(2.0 * k *
                              lambertWlog(logK / k + x * x / (2.0 * sigma * sigma * k) + np.log(x * x / (2.0 * sigma * sigma * k)))
                              )
        ans = ans.real
        return ans

    ans = gamma(d / 2.0 - k).expect(
        lambda x: beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
            ((sigma * np.sqrt(2.0 * x) + r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2) / (4.0 * sigma * np.sqrt(2.0 * x) * r)
        ),
        lb=0., ub=np.inf
    )
    return ans

def _compute_with_shift(d, k, sigma, logK, r):
    def inv_gK(x):
        # need to return g^-1(g(x)K)
        ans = sigma * np.sqrt(2.0 * k *
                              lambertWlog(- logK / k + x * x / (2.0 * sigma * sigma * k) + np.log(
                                  x * x / (2.0 * sigma * sigma * k)))
                              )
        ans = ans.real
        return ans

    ans = 1.0 - gamma(d / 2.0 - k).expect(
        lambda x: beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
            ((sigma * np.sqrt(2.0 * x) + r)**2 - inv_gK(sigma * np.sqrt(2.0 * x))**2) / (4.0 * sigma * np.sqrt(2.0 * x) * r)
        ),
        lb=0., ub=np.inf
    )
    return ans

def _binary_search_logK(d, k, sigma, r, pA):
    logK_lb, logK_ub = -100., 100.
    EPS = 1e-6
    while logK_ub - logK_lb > EPS:
        # print(f'  on {r} binary search logK in [{logK_lb}, {logK_ub}]')
        logK_mid = (logK_lb + logK_ub) / 2.0
        now_mass = _compute_at_origin(d, k, sigma, logK_mid, r)
        if now_mass < pA:
            logK_ub = logK_mid
        else:
            logK_lb = logK_mid
    ans = logK_ub
    return ans

def empirical_mean_norm(x):
    return np.sqrt(np.average(np.linalg.norm(x, ord=2, axis=1) ** 2))

# --------------main functions--------------

def sample_l2_vec(d, batch_size, cuda=False):
    """
        Sample L2 unit vectors
    :param d: dimension of vector
    :param batch_size: batch size
    :return: in numpy format
    """
    if cuda is False:
        v = np.random.normal(size=(batch_size, d))
        norms = linalg.norm(v, ord=2, axis=1)
        v = v / norms[:, np.newaxis]
    else:
        v = torch.randn((batch_size, d)).cuda()
        norms = v.norm(p=2, dim=1, keepdim=True)
        v = v.div(norms.expand_as(v))
    return v

def sample_linfty_vec(d, batch_size):
    """
        Sample Linfty unit vectors
    :param d: dimension of vector
    :param batch_size: batch size
    :return: in numpy format
    """
    v = np.random.uniform(-1., 1., size=(batch_size, d))
    ind = np.random.randint(low=0, high=d, size=batch_size)
    v[np.arange(0, batch_size), ind] = np.random.randint(low=0, high=2, size=batch_size) * 2.0 - 1.0
    return v

def sample_l1_vec(d, batch_size, cuda=False):
    '''Sample uniformly from the unit l1 sphere, i.e. the cross polytope.
    Stolen from Greg Yang's rs4a repo
    Inputs:
        device: 'cpu' | 'cuda' | other torch devices
        shape: a pair (batchsize, dim)
    Outputs:
        matrix of shape `shape` such that each row is a sample.
    '''
    if cuda is False:
        v = np.random.dirichlet(np.ones(d), batch_size)
        sign = np.random.randint(low=0, high=2, size=(batch_size, d)) * 2. - 1.
        v = v * sign
    else:
        dist = dirichlet.Dirichlet(torch.ones(d))
        v = dist.sample_n(batch_size).cuda()
        sign = torch.randint(low=0, high=2, size=(batch_size, d), device=torch.device('cuda')) * 2. - 1.
        v = v * sign
    return v


class Distribution(object):
    """
        Abstract Distribution class
    """

    def __init__(self, d, scale):
        """
            Initialization of params
        :param d: dimension of noise vectors
        :param scale: The scale of noise variance.
            Normalized to the noise magnitude of standrad L2 Gaussian with sigma = scale
        """
        self.d, self.scale = d, scale

    def sample(self, batch_size) -> np.ndarray:
        raise NotImplementedError

    def mean_norm(self) -> float:
        raise NotImplementedError

    def certify_radius(self, pA):
        raise NotImplementedError

    def info(self) -> str:
        raise NotImplementedError


class StandardGaussian(Distribution):
    """
        Standard L2 Gaussian
        proc exp(-||x||_2^2 / (2 sigma^2))
    """

    def __init__(self, d, scale, eps=1e-6, th=1.0):
        super(StandardGaussian, self).__init__(d, scale)
        self.sigma = scale

        self.eps = eps
        if th < 1.0 - eps:
            # means the gaussian sampler is thresholded
            # we need to figure out such threshold
            self.thres = gamma(self.d / 2.0 - self.k).ppf(self.th)
            print('thres set to:', self.thres)

    def set_th(self, th):
        if th < 1.0 - self.eps:
            # means the gaussian sampler is thresholded
            # we need to figure out such threshold
            self.thres = gamma(self.d / 2.0 - self.k).ppf(th)
            print('thres set to:', self.thres)

    def sample(self, batch_size, cuda=False):
        if not cuda:
            v = norm.rvs(size=(batch_size, self.d)) * self.sigma
        else:
            v = torch.randn((batch_size, self.d), device='cuda') * self.sigma
        return v

    def mean_norm(self):
        return self.sigma * np.sqrt(self.d)

    def info(self):
        return f'Standard Gaussian distribution with scale {self.scale} and sigma {self.sigma}'

    def certify_radius(self, pA):
        return self.sigma * norm.ppf(pA) if pA >= 0.5 else 0.0


class GeneralGaussian(Distribution):
    """
        General L2 Gaussian
        proc ||x||_2^{-2k} exp(-||x||_2^2 / (2 sigma^2))
    """

    def __init__(self, d, k, scale, eps=1e-6, th=1.0):
        super(GeneralGaussian, self).__init__(d, scale)

        assert d % 2 == 0
        assert k <= d/2 - 1
        d, k = int(d), int(k)
        self.k = k
        self.sigma = np.sqrt(d / (d - 2.0 * k)) * self.scale
        self.eps = eps
        self.th = th
        # the below tables are buffer, which stores the already computed certified radius
        self.pAtable = dict()
        self.rTable = dict()

        if self.th < 1.0 - eps:
            # means the gaussian sampler is thresholded
            # we need to figure out such threshold
            self.thres = gamma(self.d / 2.0 - self.k).ppf(self.th)
            print('thres set to:', self.thres)

    def set_th(self, th):
        if th < 1.0 - self.eps:
            # means the gaussian sampler is thresholded
            # we need to figure out such threshold
            self.th = th
            self.thres = gamma(self.d / 2.0 - self.k).ppf(th)
            # print('thres set to:', self.thres)

    def sample(self, batch_size, cuda=False):
        sigma = self.sigma

        # corrected!
        dist = gamma(self.d / 2.0 - self.k)
        if self.th >= 1.0 - self.eps:
            r_sq = dist.rvs(batch_size)
        else:
            cnt = 0
            ans = np.zeros(2 * batch_size, dtype=np.float)
            while cnt < batch_size:
                r_sq = dist.rvs(batch_size)
                now_cnt = np.sum(r_sq <= self.thres)
                ans[cnt: cnt + now_cnt] = r_sq[r_sq <= self.thres]
                cnt += now_cnt
                # print(f'{now_cnt}/{cnt}')
                # print('now batch:', r_sq)
                # print('now accum:', ans)
                # print('')
            r_sq = ans[:batch_size]
        r = np.sqrt(r_sq * (2.0 * sigma**2))

        if cuda is True:
            noises = torch.randn((batch_size, self.d)).cuda()
            norms = noises.norm(p=2, dim=1, keepdim=True)
            r = torch.tensor(r, dtype=torch.float32).cuda().reshape((batch_size, 1))
            # print(r.shape)
            noises = noises.div(norms.expand_as(noises)) * r
        else:
            noises = np.random.normal(size=(batch_size, self.d))
            norms = linalg.norm(noises, ord=2, axis=1)
            noises = noises / norms[:, np.newaxis] * r[:, np.newaxis]

        return noises

    def mean_norm(self):
        return self.sigma * np.sqrt(self.d - 2.0 * self.k)

    def info(self):
        return f'General Gaussian distribution with scale {self.scale} and sigma {self.sigma}'

    def certify_radius(self, pA):
        if pA < 0.5:
            return 0.0
        key = int((pA - 0.5) * (1.0 / self.eps))
        if abs(pA - self.pAtable.get(key, 0.0)) < self.eps:
            # found
            ans = self.rTable[key]
            return ans
        else:
            # according to Yang et al, the certified radius is upper bounded by that of standard Gaussian
            r_lb, r_ub = 0.0, self.sigma * norm.ppf(pA) * np.sqrt(1.0 - 2.0 * self.k / self.d)
            print(f'!!! pA={pA}')
            while r_ub - r_lb > self.eps:
                # if r_ub - r_lb < 0.01:
                #     break
                print(f'binary search r in [{r_lb}, {r_ub}]')
                r_mid = (r_lb + r_ub) / 2.0
                logK = _binary_search_logK(self.d, self.k, self.sigma, r_mid, pA)
                shifted_pA = _compute_with_shift(self.d, self.k, self.sigma, logK, r_mid)
                if shifted_pA > 0.5:
                    r_lb = r_mid
                else:
                    r_ub = r_mid
            ans = r_lb
            self.pAtable[key] = pA
            self.rTable[key] = r_lb
            return ans


class LinftyGaussian(Distribution):
    """
        Linfty Gaussian
        proc exp(-||x||_infty^2 / (2 beta^2))
    """

    def __init__(self, d, scale):
        super(LinftyGaussian, self).__init__(d, scale)

        self.beta = self.scale / np.sqrt(d / 3.0 + 2.0 / 3.0)

    def sample(self, batch_size):
        dist = gamma(self.d / 2.0)
        r_sq = dist.rvs(batch_size)
        r = np.sqrt(r_sq * (2.0 * self.beta**2))

        v = sample_linfty_vec(self.d, batch_size)
        v = r[:, np.newaxis] * v
        return v

    def mean_norm(self):
        return self.beta * np.sqrt(self.d * (self.d / 3.0 + 2.0 / 3.0))

    def info(self):
        return f'Linfty Gaussian distribution with scale {self.scale} and beta {self.beta}'



class LinftyGeneralGaussian(Distribution):
    """
        Linfty Gaussian
        proc exp(-||x||_infty^2 / (2 beta^2))
    """

    def __init__(self, d, k, scale, eps=1e-6, N=500000, alpha=0.001, batch=1000):
        super(LinftyGeneralGaussian, self).__init__(d, scale)

        assert d % 2 == 0
        assert k <= d/2 - 1
        d, k = int(d), int(k)
        self.k = k

        self.eps = eps
        self.N = N
        self.alpha = alpha
        self.batch = batch

        self.beta = self.scale / np.sqrt((d / 3.0 + 2.0 / 3.0) * (d - 2.0 * k) / d)
        # the below tables are buffer, which stores the already computed certified radius
        self.pAtable = dict()
        self.rTable = dict()

    def sample(self, batch_size):
        dist = gamma(self.d / 2.0 - self.k)
        r_sq = dist.rvs(batch_size)
        r = np.sqrt(r_sq * (2.0 * self.beta**2))

        v = sample_linfty_vec(self.d, batch_size)
        v = r[:, np.newaxis] * v
        return v

    def mean_norm(self):
        return self.beta * np.sqrt((self.d - 2.0 * self.k) * (self.d / 3.0 + 2.0 / 3.0))

    def info(self):
        return f'Linfty Gaussian distribution with scale {self.scale} and beta {self.beta}'

    def _sampler_for_radius(self):
        # sampler with compress
        dist = gamma(self.d / 2.0 - self.k)
        now_n = 0
        ans = list()
        while now_n < self.N:
            now_batch_size = min(self.batch, self.N - now_n)
            batch_rs = dist.rvs(now_batch_size)
            batch_rs = self.beta * np.sqrt(2) * np.sqrt(batch_rs)
            vec_samples = sample_linfty_vec(self.d, now_batch_size)
            vec_samples = np.vstack([vec_samples.max(axis=1), vec_samples.min(axis=1)]).T
            vec_samples = vec_samples * batch_rs[:, np.newaxis]
            ans.append(vec_samples)
            now_n += now_batch_size
        ans = np.concatenate(ans)
        return ans

    def _relative_density(self, samps):
        ans = np.linalg.norm(samps, ord=np.inf, axis=1)
        ans = - (ans ** 2) / (2.0 * self.beta * self.beta) - 2.0 * self.k * np.log(ans)
        return ans

    def _binary_search_logK_MC(self, zip_p, r_mid, pA):
        ps = self._relative_density(zip_p)
        pshifts = self._relative_density(zip_p - np.array([r_mid, r_mid])[np.newaxis, :])
        M = max(max(ps) - min(pshifts), min(ps) - max(pshifts)) + 1.
        logK_l, logK_r = -M, +M
        while logK_r - logK_l > self.eps:
            logK_mid = (logK_l + logK_r) / 2.0
            pmin, pnow = confint(np.sum(ps >= pshifts + logK_mid), self.N, self.alpha)
            # print(f"  [{logK_l}, {logK_r}] p [{pmin}, {pnow}]")
            if pnow < pA:
                logK_r = logK_mid
            else:
                logK_l = logK_mid
        return logK_r

    def _compute_with_shift(self, zip_p, r, logK):
        ps = self._relative_density(zip_p)
        pshifts = self._relative_density(zip_p + np.array([r, r])[np.newaxis, :])
        ans, _ = confint(np.sum(pshifts >= ps + logK), self.N, self.alpha)
        return ans

    def certify_radius(self, pA):
        if pA < 0.5:
            return 0.0
        key = int((pA - 0.5) * (1.0 / self.eps))
        if abs(pA - self.pAtable.get(key, 0.0)) < self.eps:
            # found
            ans = self.rTable[key]
            return ans
        else:
            # gaussian's radius provides a very large upper bound
            r_lb, r_ub = 0.0, self.scale * norm.ppf(pA)
            print(f'!!! pA={pA}')

            zip_p = self._sampler_for_radius()

            while r_ub - r_lb > self.eps:
                # print(f'binary search r in [{r_lb}, {r_ub}]')
                r_mid = (r_lb + r_ub) / 2.0
                logK = self._binary_search_logK_MC(zip_p, r_mid, pA)
                shifted_pA = self._compute_with_shift(zip_p, r_mid, logK)
                if shifted_pA > 0.5:
                    r_lb = r_mid
                else:
                    r_ub = r_mid

            ans = r_lb
            self.pAtable[key] = pA
            self.rTable[key] = r_lb
            return ans



class L1GeneralGaussian(Distribution):
    """
        Linfty Gaussian

        proc exp(-||x||_infty^2 / (2 beta^2))
    """

    def __init__(self, d, k, scale, eps=1e-6, N=50000, alpha=0.001, batch=1000):
        super(L1GeneralGaussian, self).__init__(d, scale)

        assert d % 2 == 0
        assert k <= d/2 - 1
        d, k = int(d), int(k)
        self.k = k

        self.eps = eps
        self.N = N
        self.alpha = alpha
        self.batch = batch

        self.beta = self.scale * np.sqrt(self.d * (self.d + 1) / (2.0 * (self.d - 2.0 * self.k)))
        # the below tables are buffer, which stores the already computed certified radius
        self.pAtable = dict()
        self.rTable = dict()

        self.vec_samples = None

    def sample(self, batch_size, cuda=False):
        dist = gamma(self.d / 2.0 - self.k)
        r_sq = dist.rvs(batch_size)
        r = np.sqrt(r_sq * (2.0 * self.beta**2))

        v = sample_l1_vec(self.d, batch_size, cuda)
        if cuda is False:
            v = r[:, np.newaxis] * v
        else:
            r = torch.tensor(r, dtype=torch.float32).cuda().reshape((batch_size, 1))
            v = v * r
        return v

    def mean_norm(self):
        return self.beta * np.sqrt(2.0 * (self.d - 2.0 * self.k) / (self.d + 1))

    def info(self):
        return f'L1 Gaussian distribution with scale {self.scale} and beta {self.beta}'

    def _sampler_for_radius(self):
        if self.vec_samples is None:
            # sampler
            dist = gamma(self.d / 2.0 - self.k)
            now_n = 0
            ans = list()
            while now_n < self.N:
                now_batch_size = min(self.batch, self.N - now_n)
                batch_rs = dist.rvs(now_batch_size)
                batch_rs = self.beta * np.sqrt(2) * np.sqrt(batch_rs)
                vec_samples = sample_l1_vec(self.d, now_batch_size)
                # vec_samples = np.vstack([vec_samples.max(axis=1), vec_samples.min(axis=1)]).T
                vec_samples = vec_samples * batch_rs[:, np.newaxis]
                ans.append(vec_samples)
                now_n += now_batch_size
                print(now_n)
            ans = np.concatenate(ans)
            self.vec_samples = ans
        return self.vec_samples

    def _relative_density(self, samps):
        ans = np.linalg.norm(samps, ord=1, axis=1)
        ans = - (ans ** 2) / (2.0 * self.beta * self.beta) - 2.0 * self.k * np.log(ans)
        return ans

    def _binary_search_logK_MC(self, samps, r_mid, pA):
        ps = self._relative_density(samps)
        pshifts = self._relative_density(samps - (np.ones(self.d) * r_mid)[np.newaxis, :])
        M = max(max(ps) - min(pshifts), min(ps) - max(pshifts)) + 1.
        logK_l, logK_r = -M, +M
        while logK_r - logK_l > self.eps:
            logK_mid = (logK_l + logK_r) / 2.0
            pmin, pnow = confint(np.sum(ps >= pshifts + logK_mid), self.N, self.alpha)
            # print(f"  [{logK_l}, {logK_r}] p [{pmin}, {pnow}]")
            if pnow < pA:
                logK_r = logK_mid
            else:
                logK_l = logK_mid
        return logK_r

    def _compute_with_shift(self, samps, r, logK):
        ps = self._relative_density(samps)
        pshifts = self._relative_density(samps + (np.ones(self.d) * r)[np.newaxis, :])
        ans, _ = confint(np.sum(pshifts >= ps + logK), self.N, self.alpha)
        return ans

    def certify_radius(self, pA):
        if pA < 0.5:
            return 0.0
        key = int((pA - 0.5) * (1.0 / self.eps))
        if abs(pA - self.pAtable.get(key, 0.0)) < self.eps:
            # found
            ans = self.rTable[key]
            return ans
        else:
            # gaussian's radius provides a very large upper bound
            r_lb, r_ub = 0.0, self.scale * norm.ppf(pA)
            print(f'!!! pA={pA}')

            zip_p = self._sampler_for_radius()

            while r_ub - r_lb > self.eps:
                print(f'binary search r in [{r_lb}, {r_ub}]')
                r_mid = (r_lb + r_ub) / 2.0
                logK = self._binary_search_logK_MC(zip_p, r_mid, pA)
                shifted_pA = self._compute_with_shift(zip_p, r_mid, logK)
                if shifted_pA > 0.5:
                    r_lb = r_mid
                else:
                    r_ub = r_mid

            ans = r_lb
            self.pAtable[key] = pA
            self.rTable[key] = r_lb
            return ans




class MonteCarloStandardGaussian(Distribution):
    """
        Linfty Gaussian
        proc exp(-||x||_infty^2 / (2 beta^2))
    """

    def __init__(self, d, scale, eps=1e-6, N=50000, alpha=0.001, batch=1000):
        super(MonteCarloStandardGaussian, self).__init__(d, scale)

        assert d % 2 == 0
        d = int(d)

        self.eps = eps
        self.N = N
        self.alpha = alpha
        self.batch = batch

        self.beta = self.scale
        # the below tables are buffer, which stores the already computed certified radius
        self.pAtable = dict()
        self.rTable = dict()

    def sample(self, batch_size):
        v = norm.rvs(size=(batch_size, self.d)) * self.beta
        return v

    def mean_norm(self):
        return self.beta * np.sqrt(self.d)

    def info(self):
        return f'L2 Gaussian distribution with scale {self.scale} and beta {self.beta}'

    def _sampler_for_radius(self):
        # sampler
        now_n = 0
        ans = list()
        while now_n < self.N:
            now_batch_size = min(self.batch, self.N - now_n)
            vec_samples = self.sample(batch_size=now_batch_size)
            ans.append(vec_samples)
            now_n += now_batch_size
        ans = np.concatenate(ans)
        return ans

    def _relative_density(self, samps):
        ans = np.linalg.norm(samps, ord=2, axis=1)
        ans = - (ans ** 2) / (2.0 * self.beta * self.beta) #- 2.0 * self.k * np.log(ans)
        return ans

    def _binary_search_logK_MC(self, samps, r_mid, pA):
        ps = self._relative_density(samps)
        pshifts = self._relative_density(samps - (np.ones(self.d) * r_mid)[np.newaxis, :])
        M = max(max(ps) - min(pshifts), min(ps) - max(pshifts)) + 1.
        logK_l, logK_r = -M, +M
        while logK_r - logK_l > self.eps:
            logK_mid = (logK_l + logK_r) / 2.0
            pmin, pnow = confint(np.sum(ps >= pshifts + logK_mid), self.N, self.alpha)
            # print(f"  [{logK_l}, {logK_r}] p [{pmin}, {pnow}]")
            if pnow < pA:
                logK_r = logK_mid
            else:
                logK_l = logK_mid
        return logK_r

    def _compute_with_shift(self, samps, r, logK):
        ps = self._relative_density(samps)
        pshifts = self._relative_density(samps + (np.ones(self.d) * r)[np.newaxis, :])
        ans, _ = confint(np.sum(pshifts >= ps + logK), self.N, self.alpha)
        return ans

    def certify_radius(self, pA):
        if pA < 0.5:
            return 0.0
        key = int((pA - 0.5) * (1.0 / self.eps))
        if abs(pA - self.pAtable.get(key, 0.0)) < self.eps:
            # found
            ans = self.rTable[key]
            return ans
        else:
            # gaussian's radius provides a very large upper bound
            r_lb, r_ub = 0.0, self.scale * norm.ppf(pA)
            print(f'!!! pA={pA}')

            zip_p = self._sampler_for_radius()

            while r_ub - r_lb > self.eps:
                print(f'binary search r in [{r_lb}, {r_ub}]')
                r_mid = (r_lb + r_ub) / 2.0
                logK = self._binary_search_logK_MC(zip_p, r_mid, pA)
                shifted_pA = self._compute_with_shift(zip_p, r_mid, logK)
                if shifted_pA > 0.5:
                    r_lb = r_mid
                else:
                    r_ub = r_mid

            ans = r_lb
            self.pAtable[key] = pA
            self.rTable[key] = r_lb
            return ans



"""
    below is just the correctness test script
"""
if __name__ == '__main__':
    # times = 100
    # # dist = StandardGaussian(d = 3072, scale = 0.5)
    # # dist = GeneralGaussian(d = 3072, k = 1530, scale = 0.5)
    # dist = LinftyGaussian(d = 3072, scale = 0.5)
    # mean = dist.mean_norm()
    # print(mean)
    # delta = 0.
    # for time in range(times):
    #     a = dist.sample(100)
    #     empirical_mean = empirical_mean_norm(a)
    #     print('emprcl', empirical_mean)
    #     delta += empirical_mean - mean
    # delta /= times
    # print('avg delta', delta)
    #
    # # a = sample_l2_vec(d=3, batch_size=5)
    # # b = sample_linfty_vec(d=3, batch_size=5)
    # # print(a)
    # # print(b)
    # # print(empirical_mean_norm(a))
    # # print(empirical_mean_norm(b))

    D = 3072
    P = 0.99

    # dist = LinftyGeneralGaussian(d=D, k=1510, scale=0.5)
    # print(dist.certify_radius(P))

    # dist = LinftyGeneralGaussian(d=D, k=1530, scale=0.5)
    # print(dist.certify_radius(P))

    dist = StandardGaussian(d=D, scale=0.5)
    print(dist.certify_radius(P) / np.sqrt(D))

    # dist = MonteCarloStandardGaussian(d=D, scale=0.5)
    # print(dist.certify_radius(P))

    # print(dist.certify_radius(P))
    #
    # dist = GeneralGaussian(d=D, k=1530, scale=0.5)
    # print(dist.certify_radius(P))

    # k=1530, P=0.99, scale=0.5, ans = 0.01365756751308004
    # dist = L1GeneralGaussian(d=D, k=1530, scale=0.5)
    # print(dist.certify_radius(P))

    # k=1510, P=0.99, scale=0.5, ans = 0.014246600030091532
    # dist = L1GeneralGaussian(d=D, k=1510, scale=0.5)
    # print(dist.certify_radius(P))

    # k=1510, P=0.995, scale=0.5, ans = 0.015839526509555158
    dist = L1GeneralGaussian(d=D, k=1510, scale=0.5)
    print(dist.certify_radius(0.995))

    # k=1510, P=0.999, scale=0.5, ans = 0.01868666696103436
    dist = L1GeneralGaussian(d=D, k=1510, scale=0.5)
    print(dist.certify_radius(0.999))