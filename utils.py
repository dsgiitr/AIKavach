import scipy
import numpy as np



def lambertWlog(logx, mode='lowerbound'):
    # compute lambertW(x) with log x as the input
    # stolen from Greg Yang et al's rs4a @ GitHub
    z = logx
    '''Computes LambertW(e^z) numerically safely.
    For small value of z, we use `scipy.special.lambertw`.
    For large value of z, we apply the approximation

        z - log(z) < W(e^z) < z - log(z) - log(1 - log(z)/z).
    '''
    if z > 500:
        if mode == 'lowerbound':
            return z - np.log(z)
        elif mode == 'upperbound':
            return z - np.log(z) - np.log(1 - np.log(z) / z)
        else:
            return np.NaN
            # raise ValueError('Unknown mode: ' + str(mode))
    else:
        return scipy.special.lambertw(np.exp(z)).real


# def calc_adaptive_sigma(sigma:float, d:int, k:int):
#     return sigma * np.sqrt((d + 2) / (d + 2 - 2 * k))


def read_pAs(file_path):
    """
        The format of sampling file:
        line could be empty, start with x, or start with o
        If start with x, metadata
        If start with o, then it will follow three numbers: #no, pA lower bound, pA upper bound
        Here, the pA is the probability of the true class.
    :param file_path:
    :return:
    """
    arr = list()
    no_set = set()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(' ')
            if len(fields) > 0 and fields[0] == 'o':
                no, pAL, pAU = fields[1:]
                no, pAL, pAU = int(no), float(pAL), float(pAU)
                if no not in no_set:
                    arr.append((no, pAL, pAU))
                    no_set.add(no)
    return arr

def read_orig_Rs(file_path, num_stds):
    """
        The format of original R file:
        Each line corresponds to a sample.
    :param file_path:
    :param aux_stds:
    :return: [instance_no, radius, p1low, p1high, [[other-p1low1, other-p1high1], ..., [other-p1lowN, other-p1highN]]]
    """
    res = list()
    res_in_dict = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                fields = line.strip().split(' ')
                no, r = int(fields[0]), float(fields[1])
                cur_line = [no, r, None, None, [[None, None] for _ in range(len(num_stds))]]
                res_in_dict[no] = cur_line
    for i in sorted(res_in_dict.keys()):
        res.append(res_in_dict[i])
    return res