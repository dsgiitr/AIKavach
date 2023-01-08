import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time, sleep
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian
import smooth
from algo import calc_fast_beta_th, check
import numpy as np
from random import random
from multiprocessing.pool import Pool, ThreadPool
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from icecream import ic

workers = 10
ORIG_R_EPS = 5e-5
DUAL_EPS = 1e-8

class FinishedModel():
    def __init__(self, denoised_model, d, k, num_classes, dist_1, dist_2, std_1, std_2, alpha, num_sampling_min = 20) -> None:
        self.num_sampling_min = num_sampling_min
        self.denoised_model = denoised_model
        self.denoised_model.eval()
        # the line below is temporary until denoiser stuff is integrated
        self.num_dims = d
        self.alpha = alpha
        self.k = k
        self.num_classes = num_classes
        self.dist_name_1 = dist_1
        self.dist_name_2 = dist_2
        self.std_1 = std_1
        self.std_2 = std_2
        if dist_1 == 'gaussian':
            self.dist_1 = StandardGaussian(self.num_dims, std_1, eps = 0.01)
        elif dist_1 == 'general-gaussian':
            self.dist_1 = GeneralGaussian(self.num_dims, self.k, std_1, th = 1.0, eps = 0.01)
        elif dist_1 == 'infty-gaussian':
            self.dist_1 = LinftyGaussian(self.num_dims, std_1, eps = 0.01)
        elif dist_1 == 'infty-general-gaussian':
            self.dist_1 = LinftyGeneralGaussian(self.num_dims, self.k, std_1, eps = 0.01)
        elif dist_1 == 'L1-general-gaussian':
            self.dist_1 = L1GeneralGaussian(self.num_dims, self.k, std_1, eps = 0.01)
        else:
            raise NotImplementedError('Unsupported smoothing distribution')

        if dist_2 == 'gaussian':
            self.dist_2 = StandardGaussian(self.num_dims, std_2, eps = 0.01)
        elif dist_2 == 'general-gaussian':
            self.dist_2 = GeneralGaussian(self.num_dims, self.k, std_2, th = 1.0, eps = 0.01)
        elif dist_2 == 'infty-gaussian':
            self.dist_2 = LinftyGaussian(self.num_dims, std_2, eps = 0.01)
        elif dist_2 == 'infty-general-gaussian':
            self.dist_2 = LinftyGeneralGaussian(self.num_dims, self.k, std_2, eps = 0.01)
        elif dist_2 == 'L1-general-gaussian':
            self.dist_2 = L1GeneralGaussian(self.num_dims, self.k, std_2, eps = 0.01)
        else:
            raise NotImplementedError('Unsupported smoothing distribution')

    def label_inference_without_certification(self, x, num_sampling, fractional_loss, batch_size = 1):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA1_1, realN_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_1, realN_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1, self.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2, self.alpha)
        num_opt_1 = self.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.std_1, self.alpha)
        num_opt_2 = self.get_opt_num_sampling(p2low_2, p2high_2, num_sampling, fractional_loss, batch_size, self.std_2, self.alpha)
        nA1_2, realN_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = num_opt_1, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_2, realN_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = num_opt_2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA_1 = nA1_1 + nA1_2
        nA_2 = nA2_1 + nA2_2
        nA = nA_1 + nA_2
        return nA.argmax()

    def logits_inference_without_certification(self, x, num_sampling, fractional_loss, batch_size = 1):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA1_1, realN_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_1, realN_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1, self.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2, self.alpha)
        num_opt_1 = self.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.std_1, self.alpha)
        num_opt_2 = self.get_opt_num_sampling(p2low_2, p2high_2, num_sampling, fractional_loss, batch_size, self.std_2, self.alpha)
        nA1_2, realN_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = num_opt_1, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_2, realN_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = num_opt_2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA_1 = nA1_1 + nA1_2
        nA_2 = nA2_1 + nA2_2
        nA = nA_1 + nA_2
        return F.softmax(torch.Tensor(nA)).detach().numpy()

    # def get_pAlist(self, nA, realN, local_alpha):

    def orig_radius_pool_func(self, args, dist):
        """
        Paralleled original radius computing function
        :param args:
        :return:
        """
        pA = args
        stime = time()
        r = dist.certify_radius(pA)
        return r, time() - stime

    def new_radius_pool_func(self, args):
        no, orig_r, pAsigmaL, pAsigmaR, pAbetaL, pAbetaR = args
        print('On #', no)
        stime = time()
    
        new_r = orig_r
        if (orig_r <= 1e-5 and pAsigmaL <= 0.1 and pAbetaL <= 0.1) or pAbetaL is None or pAbetaR is None:
            pass
        else:
            if orig_r <= 1e-5:
                print('try even though orig_r is 0')
            if bunk_radius_mode == 'grid':
                slot = int(orig_r / bunk_radius_unit)
                new_r = orig_r
                while True:
                    slot += 1
                    try:
                        # ! suppress possible exceptions...
                        print(f'check rad = {slot * bunk_radius_unit} for pA = {pAsigmaL} (old rad = {orig_r})')
                        if check(bunk_disttype, slot * bunk_radius_unit,
                                 bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                                 pAsigmaL, pAsigmaR, pAbetaL, pAbetaR,
                                 eps=DUAL_EPS):
                            new_r = bunk_radius_unit * slot
                            print(f'  #{no} New r = {new_r}')
                        else:
                            break
                    except Exception as e:
                        # print(type(e))
                        print('exception encountered')
                        print(e)
                        break
                        # raise e
            else:
                r_delta = bunk_radius_eps
                while True:
                    print(f'  #{no} Check radius +', r_delta)
                    if r_delta > 50.0 * orig_r and orig_r >= 0.1:
                        # I don't quite believe DSRS can improve over 5000% in practice (though theoretically can as described in our paper)
                        raise Exception(f'Suspected numerical error @ #{no} with orig R = {orig_r}, pA in [{pAsigmaL}, {pAsigmaR}], pAbeta in [{pAbetaL}, {pAbetaR}]')
                    if check(bunk_disttype, orig_r + r_delta,
                             bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                             pAsigmaL, pAsigmaR, pAbetaL, pAbetaR, eps=DUAL_EPS):
                        r_delta *= 2.
                    else:
                        r_delta /= 2.
                        break
                if r_delta >= bunk_radius_eps:
                    new_r = orig_r + r_delta
                if bunk_radius_mode == 'precise':
                    rad_L, rad_R = orig_r + r_delta, orig_r + 2. * r_delta
                    while rad_R - rad_L > bunk_radius_eps:
                        mid = (rad_L + rad_R) / 2.0
                        res = check(bunk_disttype, mid,
                                    bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                                    pAsigmaL, pAsigmaR, pAbetaL, pAbetaR, eps=DUAL_EPS)
                        if res:
                            rad_L = mid
                        else:
                            rad_R = mid
                    new_r = rad_L
        print(f'Result on #{no} (sigma={bunk_radius_sigma}, beta={bunk_radius_beta}) R = {orig_r} + {new_r - orig_r} [time={time() - stime} s]')
        runtime = time() - stime
        # avoid conflict on global writer
        sleep(random())
        return no, new_r, new_r - orig_r, ((new_r - orig_r) / max(orig_r, 1e-5)), runtime


    def bunk_radius_calc(self, full_info, disttype, d, k, sigma, betas, mode, unit=0.05, eps=0.01):
        """
        The entrance function, or the dispatcher, for the improved radius computation
        :param full_info: the full info list, [[no, radius, p1low, p1high, [[p2low, p2high], ...]]
        :param result_dir: the directory to save the result
        :param d: input dimension
        :param k: for general Gaussian, parameter k; for others, it is None
        :param sigma: variance scaling
        :param betas: for general Gaussian or standard Gaussian, the list of betas to derive improved radius; for others, it is an empty list
        :param N: number of samples
        :param alpha: confidence level
        :param mode: must be 'grid'/'fast'/'precise'
        :param unit: grid search granularity
        :param eps: precision control
        :return: None
        """
        global bunk_disttype, bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta
        global bunk_radius_mode, bunk_radius_unit, bunk_radius_eps
        
        bunk_disttype = disttype
        bunk_radius_d, bunk_radius_k = d, k
        bunk_radius_mode, bunk_radius_unit, bunk_radius_eps = mode, unit, eps
        if bunk_disttype == 'general-gaussian' or bunk_disttype == 'gaussian' \
                or bunk_disttype == 'general-gaussian-th' or bunk_disttype == 'gaussian-th':
            for i, beta in enumerate([betas]):
                print(f'Now on beta = {beta}')
                # compute the real beta
                if bunk_disttype == 'general-gaussian':
                    bunk_radius_sigma = np.sqrt(d / (d - 2.0 * k)) * sigma
                    bunk_radius_beta = np.sqrt(d / (d - 2.0 * k)) * beta
                elif bunk_disttype == 'general-gaussian-th':
                    bunk_radius_sigma = np.sqrt(d / (d - 2.0 * k)) * sigma
                    bunk_radius_beta = beta
                else:
                    bunk_radius_sigma = sigma
                    bunk_radius_beta = beta
                print(f"full info: {full_info}")
                view = [full_info[0], full_info[1], full_info[2], full_info[3], full_info[4][0][0], full_info[4][0][1]]
                res = self.new_radius_pool_func(view)
        _, r, __, ___, ____ = res
        return r

    def find_opt_batchnum(self, iss, pa_lower, pa_upper):
        list_p = list(iss.keys())
        pa_lower = np.clip(pa_lower, 0.0, 1.0)
        pa_upper = np.clip(pa_upper, 0.0, 1.0)
        for i, p in enumerate(list_p):
            if pa_lower <= p:
                opt_batchnum = max(iss[list_p[max(0,i - 1)]], iss[p])
                break
        for i, p in enumerate(list_p):
            if pa_upper <= p:
                opt_batchnum = max(opt_batchnum, iss[list_p[max(0,i - 1)]], iss[p])
                break
        return opt_batchnum

    def _lower_confidence_bound(self, NA, N, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


    def generate_iss(self, loss, batch_size, upper, sigma, alpha) -> dict:
        iss = {}
        max_sample_size = upper * batch_size
        for pa in list(np.arange(500 + 1) * 0.001+0.5):
            iss[pa] = upper
            opt_radius = sigma * norm.ppf(self._lower_confidence_bound(max_sample_size * pa, max_sample_size, alpha))
            standard = opt_radius*(1- loss)
            if standard <= 0:
                iss[pa] = 0
            else:
                for num in range(upper + 1):
                    sample_size = num * batch_size
                    if sigma * norm.ppf(self._lower_confidence_bound(sample_size * pa, sample_size, alpha)) >= standard:
                        iss[pa] = num
                        break
        return iss

    def get_opt_num_sampling(self, plow, phigh, n_max, fractional_loss_in_radius, batch_size, sigma, alpha):
        iss = self.generate_iss(fractional_loss_in_radius, batch_size, n_max//batch_size, sigma, alpha)
        opt = self.find_opt_batchnum(iss, plow, phigh)
        return opt

    def inference_and_certification(self, x, num_sampling, fractional_loss, batch_size = 1):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA1_1, realN_1_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_1, realN_2_1 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = self.num_sampling_min // 2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2_1, self.alpha)
        num_opt_1 = self.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.std_1, self.alpha)
        num_opt_2 = self.get_opt_num_sampling(p2low_2, p2high_2, num_sampling, fractional_loss, batch_size, self.std_2, self.alpha)
        nA1_2, realN_1_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_1, num = num_opt_1, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA2_2, realN_2_2 = smooth.get_logits(model = self.denoised_model, x = x, dist = self.dist_2, num = num_opt_2, num_classes = self.num_classes, batch_size = batch_size, num_crop = 5)
        nA_1 = nA1_1 + nA1_2
        nA_2 = nA2_1 + nA2_2
        nA = nA_1 + nA_2
        realN_1 = realN_1_1 + realN_1_2
        realN_2 = realN_2_1 + realN_2_2
        p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, self.alpha)
        p1low_2, p1high_2 = smooth.confidence_bound(nA_2[nA_1.argmax().item()].item(), realN_2, self.alpha)
        now_r, now_time = self.orig_radius_pool_func(p1high_1, self.dist_1)
        full_info = [0, now_r, p1low_1, p1high_1, [[p1low_2, p1high_2]]]
        r = self.bunk_radius_calc(full_info, self.dist_name_2, self.num_dims, self.k, self.std_1, self.std_2, 'fast')
        return F.softmax(torch.Tensor(nA)).detach().numpy(), r

class ChotaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28*28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512,512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512,10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)
        
    def forward(self, x):
        # flatten image input
        x = x.view(-1,28*28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
         # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = ChotaModel()
    secure_model = FinishedModel(model, 784, 380, 10, 'general-gaussian', 'general-gaussian', 0.5, 0.4, 0.0005, num_sampling_min = 100)
    x = torch.randn((28, 28)).float()
    label = secure_model.label_inference_without_certification(x, 1_000, 0.01, batch_size = 64)
    logits_old = secure_model.logits_inference_without_certification(x, 1000, 0.01, batch_size = 64)
    logits, r = secure_model.inference_and_certification(x, 100, 0.01, batch_size = 64)
    ic(label)
    ic(logits_old)
    print(f"logits are: {logits}, radius is {r}")
    print('meow meow') 