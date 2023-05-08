from packaging_class import FinishedModel, ChotaModel
import unittest
import torch
import torchvision.transforms as transforms
import numpy
import smooth
from icecream import ic
from time import time
import torchvision
import matplotlib.pyplot as plt
import warnings


def test_num_samples(model,x):
    num_sampling = 10000
    fractional_loss = 0.02
    batch_size = 64
    nA1_1, realN_1_1 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = model.num_sampling_min, num_classes = model.num_classes, batch_size = batch_size, num_crop = 1)
    p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, model.alpha)
    k_hat = []
    k_bar = []
    for i in range(0,num_sampling,10):
        num_opt = model.get_opt_num_sampling(p1low_1, p1high_1, i, fractional_loss, batch_size, model.std_1, model.alpha)
        print(i,num_opt)
        k_hat.append(i)
        k_bar.append(num_opt)
    plt.plot(k_hat,k_bar)
    plt.figure(figsize=(8,8))
    plt.show()
    plt.savefig('test1.png')

def test_og_radius(model,x):
    num_sampling = 20000
    fractional_loss = 0.02
    batch_size = 100
    nA_1, realN_1 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = model.num_sampling_min, num_classes = model.num_classes, batch_size = batch_size)
    p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, model.alpha)
    r1,_ = model.orig_radius_pool_func(p1low_1, model.dist_1)
    num_opt = model.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, model.std_1, model.alpha)
    nA_2, realN_2 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = num_opt*batch_size - model.num_sampling_min, num_classes = model.num_classes, batch_size = batch_size)
    nA = nA_1 + nA_2
    realN = realN_1 + realN_2
    p1low_2, p1high_2 = smooth.confidence_bound(nA[nA.argmax().item()].item(), realN, model.alpha)
    r2,_ = model.orig_radius_pool_func(p1low_2, model.dist_1)
    nA_max, realN_max = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = num_sampling, num_classes = model.num_classes, batch_size = batch_size)
    p1low_m, p1high_m = smooth.confidence_bound(nA_max[nA_max.argmax().item()].item(), realN_max, model.alpha)
    r_max,_ = model.orig_radius_pool_func(p1low_m, model.dist_1)
    print(realN)
    print('Radius with minimum sampling : ',r1)
    print('Radius with optimal sampling : ',r2)
    print('Radius with maximum sampling : ',r_max)



def test_iss_dsrs(model,x):
    num_sampling = 10000
    batch_size = 64
    fractional_loss = 0.02
    nA_1_1, realN_1_1 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = model.num_sampling_min, num_classes = model.num_classes, batch_size = batch_size)
    p1low_1, p1high_1 = smooth.confidence_bound(nA_1_1[nA_1_1.argmax().item()].item(), realN_1_1, model.alpha)
    num_opt = model.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, model.std_1, model.alpha)
    nA_1_2, realN_1_2 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num = num_opt*batch_size - model.num_sampling_min, num_classes = model.num_classes, batch_size = batch_size)
    nA_1 = nA_1_1 + nA_1_2
    realN_1 = realN_1_1 + realN_1_2
    p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, model.alpha)
    nA_2, realN_2 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_2, num = num_opt*batch_size, num_classes = model.num_classes, batch_size = batch_size)
    p2low_2, p2high_2 = smooth.confidence_bound(nA_2[nA_2.argmax().item()].item(), realN_2, model.alpha)
    r1, now_time = model.orig_radius_pool_func(p1low_1, model.dist_1)
    full_info = [0, r1, p1low_1, p1high_1, [[p2low_2, p2high_2]]]
    r2 = model.bunk_radius_calc(full_info, model.dist_name_2, model.num_dims, model.k, model.std_1, model.std_2, 'fast')
    return r2

def test_dsrs(model,x):
    num_sampling = 10000
    batch_size = 64
    nA1_1, realN_1_1 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_1, num =  num_sampling//2, num_classes = model.num_classes, batch_size = batch_size)
    nA2_1, realN_2_1 = smooth.get_logits(model = model.denoised_model, x = x, dist = model.dist_2, num = num_sampling//2, num_classes = model.num_classes, batch_size = batch_size)
    p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, model.alpha)
    p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2_1, model.alpha)
    r1, now_time = model.orig_radius_pool_func(p1low_1, model.dist_1)
    full_info = [0, r1, p1low_1, p1high_1, [[p2low_2, p2high_2]]]
    r2 = model.bunk_radius_calc(full_info, model.dist_name_2, model.num_dims, model.k, model.std_1, model.std_2, 'fast')
    return r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    denoised = torch.load('../data/model.pt')
    model = FinishedModel(denoised, 784, 380, 10, 'general-gaussian', 'general-gaussian', 0.5, 0.4, 0.0005, num_sampling_min = 100)
    train_dataset = torchvision.datasets.MNIST(root='../data',
                                             train=True,                                             
                                             transform = transforms.ToTensor(),
                                             download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=1,
                                            shuffle=True)
    
    x, _ = next(iter(train_loader))
    test_og_radius(model,x)
    # start_time = time()
    # r1 = test_iss_dsrs(model,x)
    # t1 = time() - start_time
    # r2 = test_dsrs(model,x)
    # t2 = time() - t1
    # print("With iss, radius : {}, time = {}".format(r1,t1))
    # print("Without iss, radius : {}, time = {}".format(r2,t2))