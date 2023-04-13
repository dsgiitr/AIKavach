from packaging_class import FinishedModel, ChotaModel
import unittest
import torch
import torchvision.transforms as transforms
import numpy
import smooth
from icecream import ic
import torchvision


class TestFinishedModel(unittest.TestCase):
    def setUp(self):
        self.denoised = ChotaModel()
        self.model = FinishedModel(self.denoised, 784, 380, 10, 'general-gaussian', 'general-gaussian', 0.5, 0.4, 0.0005, num_sampling_min = 100)
        self.x = torch.randn((28, 28)).float()
    def LabelInference(self):
        num_sampling = 1
        fractional_loss = 0.5
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num =  num_sampling, num_classes = self.model.num_classes, batch_size = batch_size)
        nA2_1, realN_2_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_2, num = num_sampling, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2_1, self.model.alpha)
        now_r, now_time = self.model.orig_radius_pool_func(p1high_1, self.model.dist_1)
        full_info = [0, now_r, p1low_1, p1high_1, [[p2low_2, p2high_2]]]
        r1 = self.model.bunk_radius_calc(full_info, self.model.dist_name_2, self.model.num_dims, self.model.k, self.model.std_1, self.model.std_2, 'fast')
        num_opt_1 = self.model.get_opt_num_sampling(p1low_1, p1high_1, realN_1_1, fractional_loss, batch_size, self.model.std_1, self.model.alpha)
        num_opt_2 = self.model.get_opt_num_sampling(p2low_2, p2high_2, realN_2_1, fractional_loss, batch_size, self.model.std_2, self.model.alpha)
        nA1_2, realN_1_2 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = num_opt_1, num_classes = self.model.num_classes, batch_size = batch_size)
        nA2_2, realN_2_2 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_2, num = num_opt_2, num_classes = self.model.num_classes, batch_size = batch_size)
        nA_1 = nA1_1 + nA1_2
        nA_2 = nA2_1 + nA2_2
        nA = nA_1 + nA_2
        realN_1 = realN_1_1 + realN_1_2
        realN_2 = realN_2_1 + realN_2_2
        p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, self.model.alpha)
        p1low_2, p1high_2 = smooth.confidence_bound(nA_2[nA_1.argmax().item()].item(), realN_2, self.model.alpha)
        now_r, now_time = self.model.orig_radius_pool_func(p1high_1, self.model.dist_1)
        full_info = [0, now_r, p1low_1, p1high_1, [[p1low_2, p1high_2]]]
        r2 = self.model.bunk_radius_calc(full_info, self.model.dist_name_2, self.model.num_dims, self.model.k, self.model.std_1, self.model.std_2, 'fast')
        self.assertGreater(r2,r1)
    
    def radius(self):
        num_sampling = 1000
        fractional_loss = 0.05
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = num_sampling, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        r1, _ = self.model.orig_radius_pool_func(p1high_1, self.model.dist_1)
        num_opt_1 = self.model.get_opt_num_sampling(p1low_1, p1high_1, realN_1_1, fractional_loss, batch_size, self.model.std_1, self.model.alpha)
        nA1_2, realN_1_2 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = num_opt_1, num_classes = self.model.num_classes, batch_size = batch_size)
        nA_1 = nA1_1 + nA1_2
        realN_1 = realN_1_1 + realN_1_2
        p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, self.model.alpha)
        r2, _ = self.model.orig_radius_pool_func(p1high_1,self.model.dist_1)
        self.assertGreater(r2,r1)

    def num_samples(self):
        num_sampling = 1000
        fractional_loss = 0.02
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = self.model.num_sampling_min//2, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        num_opt = self.model.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.model.std_1, self.model.alpha)
        self.assertGreater(num_opt,0)
    
    def test_bunk_radius(self):
        num_sampling = 10000
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num =  num_sampling//2, num_classes = self.model.num_classes, batch_size = batch_size)
        nA2_1, realN_2_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_2, num = num_sampling//2, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2_1, self.model.alpha)
        r1, now_time = self.model.orig_radius_pool_func(p1high_1, self.model.dist_1)
        full_info = [0, r1, p1low_1, p1high_1, [[p2low_2, p2high_2]]]
        r2 = self.model.bunk_radius_calc(full_info, self.model.dist_name_2, self.model.num_dims, self.model.k, self.model.std_1, self.model.std_2, 'precise')
        self.assertGreater(r2,r1)


if __name__ == "__main__":
    unittest.main()