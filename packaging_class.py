import torch
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian
import smooth

class FinishedModel():
    def __init__(self, raw_model, denoiser, d, num_classes, dist_1, dist_2, std_1, std_2) -> None:
        self.model = raw_model
        self.model.eval()
        self.denoiser = denoiser
        self.denoiser.eval()
        self.stacked_model = self.model(self.denoiser)
        self.num_dims = d
        self.num_classes = num_classes
        if dist_1 == 'gaussian':
            self.dist_1 = StandardGaussian(self.num_dims, std_1)
        elif dist_1 == 'general-gaussian':
            self.dist_1 = GeneralGaussian(self.num_dims, std_2, std_1, th = 1.0)
        elif dist_1 == 'infty-gaussian':
            self.dist_1 = LinftyGaussian(self.num_dims, std_1)
        elif dist_1 == 'infty-general-gaussian':
            self.dist_1 = LinftyGeneralGaussian(self.num_dims, std_2, std_1)
        elif dist_1 == 'L1-general-gaussian':
            self.dist_1 = L1GeneralGaussian(self.num_dims, std_2, std_1)
        else:
            raise NotImplementedError('Unsupported smoothing distribution')

        if dist_2 == 'gaussian':
            self.dist_2 = StandardGaussian(self.num_dims, std_2)
        elif dist_2 == 'general-gaussian':
            self.dist_2 = GeneralGaussian(self.num_dims, std_2, std_2, th = 1.0)
        elif dist_2 == 'infty-gaussian':
            self.dist_2 = LinftyGaussian(self.num_dims, std_2)
        elif dist_2 == 'infty-general-gaussian':
            self.dist_2 = LinftyGeneralGaussian(self.num_dims, std_2, std_2)
        elif dist_2 == 'L1-general-gaussian':
            self.dist_2 = L1GeneralGaussian(self.num_dims, std_2, std_2)
        else:
            raise NotImplementedError('Unsupported smoothing distribution')

    def label_inference_without_certification(self, x, num_sampling):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA_1, realN_1 = smooth.sample_noise(self.stacked_model, x, self.dist_1, num_sampling // 2, self.num_classes, x[0])
        nA_2, realN_2 = smooth.sample_noise(self.stacked_model, x, self.dist_2, num_sampling // 2, self.num_classes, x[0])
        nA = nA_1 + nA_2
        return nA.argmax(1)

    def logits_inference_without_certification(self, x, num_sampling):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA_1, realN_1 = smooth.sample_noise(self.stacked_model, x, self.dist_1, num_sampling // 2, self.num_classes, x[0])
        nA_2, realN_2 = smooth.sample_noise(self.stacked_model, x, self.dist_2, num_sampling // 2, self.num_classes, x[0])
        nA = nA_1 + nA_2
        return torch.nn.functional.softmax(nA)

    def inference_and_certification(self, x, num_sampling):
        """""
        Expects x's dimensions to be in the order N, C, H, W
        """""
        nA_1, realN_1 = smooth.sample_noise(self.stacked_model, x, self.dist_1, num_sampling // 2, self.num_classes, x[0])
        nA_2, realN_2 = smooth.sample_noise(self.stacked_model, x, self.dist_2, num_sampling // 2, self.num_classes, x[0])
        nA = nA_1 + nA_2
        p1low, p1high = smooth.confidence_bound(nA, realN, local_alpha)
        
        return torch.nn.functional.softmax(nA)