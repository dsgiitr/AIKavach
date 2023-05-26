from packaging_class import FinishedModel, ChotaModel
import unittest
import torch
import torchvision.transforms as transforms
import numpy
import smooth
#from icecream import ic
import torchvision
import warnings


class TestFinishedModel(unittest.TestCase):
    def setUp(self):
        self.denoised = ChotaModel()
        self.model = FinishedModel(self.denoised, 784, 380, 10, 'general-gaussian', 'general-gaussian', 0.5, 0.4, 0.0005, num_sampling_min = 100)
        self.x = torch.randn((28, 28)).float()

    def test_num_samples(self):
        num_sampling = 20000
        fractional_loss = 0.02
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = self.model.num_sampling_min, num_classes = self.model.num_classes, batch_size = batch_size, num_crop=1)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        num_opt = self.model.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.model.std_1, self.model.alpha)
        self.assertGreater(num_opt,0) 

    def test_iss_radius(self):
        num_sampling = 20000
        fractional_loss = 0.02
        batch_size = 64
        nA_1, realN_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = self.model.num_sampling_min, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, self.model.alpha)
        r1,_ = self.model.orig_radius_pool_func(p1low_1, self.model.dist_1)
        num_opt = self.model.get_opt_num_sampling(p1low_1, p1high_1, num_sampling, fractional_loss, batch_size, self.model.std_1, self.model.alpha)
        nA_2, realN_2 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num = num_opt*batch_size - self.model.num_sampling_min, num_classes = self.model.num_classes, batch_size = batch_size)
        nA = nA_1 + nA_2
        realN = realN_1 + realN_2
        p1low_2, p1high_2 = smooth.confidence_bound(nA[nA.argmax().item()].item(), realN, self.model.alpha)
        r2,_ = self.model.orig_radius_pool_func(p1low_2, self.model.dist_1)
        self.assertGreater(r2,r1)
        
    def test_bunk_radius(self):
        num_sampling = 20000
        batch_size = 64
        nA1_1, realN_1_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_1, num =  num_sampling//2, num_classes = self.model.num_classes, batch_size = batch_size)
        nA2_1, realN_2_1 = smooth.get_logits(model = self.model.denoised_model, x = self.x, dist = self.model.dist_2, num = num_sampling//2, num_classes = self.model.num_classes, batch_size = batch_size)
        p1low_1, p1high_1 = smooth.confidence_bound(nA1_1[nA1_1.argmax().item()].item(), realN_1_1, self.model.alpha)
        p2low_2, p2high_2 = smooth.confidence_bound(nA2_1[nA2_1.argmax().item()].item(), realN_2_1, self.model.alpha)
        r1, now_time = self.model.orig_radius_pool_func(p1low_1, self.model.dist_1)
        full_info = [0, r1, p1low_1, p1high_1, [[p2low_2, p2high_2]]]
        r2 = self.model.bunk_radius_calc(full_info, self.model.dist_name_2, self.model.num_dims, self.model.k, self.model.std_1, self.model.std_2, 'precise')
        self.assertGreater(r2,r1)

    def denoiser(self):
        transform = transforms.Resize((28,28))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_sampling = 20000
        batch_size = 64
        fractional_loss = 0.02
        model = self.model
        de = denoiser(in_nc=2,out_nc=1,nc = [64,128,256,512],nb=4)
        de.ld('drunet_gray.pth')
        noise_level_img = 15
        noise_level_model = 15
        y = torch.randn((32, 32, 1)).float()
        img_L = util.uint2single(y)
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
        img_L = util.single2tensor4(img_L)
        x = img_L.to(device)
        img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        x_denoised = de.drunet.netG.forward(img_L)
        _,r1 = model.inference_and_certification(transform(x),num_sampling, fractional_loss, batch_size)
        _,r2 = model.inference_and_certification(transform(x_denoised),num_sampling, fractional_loss, batch_size)
        self.assertGreater(r2,r1)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    unittest.main()