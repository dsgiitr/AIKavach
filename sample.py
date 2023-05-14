import torch
import torch.nn as nn
from packaging_class import FinishedModel
import argparse

class DenoisedModel(nn.Module):
    def __init__(self, denoiser_path, classifier_path):
        super().__init__()
        self.denoiser = torch.load(denoiser_path)
        self.classifier = torch.load(classifier_path)

    def forward(self, x):
        return self.classifier(self.denoiser(x))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser_path', type=str)
    parser.add_argument('--classifier_path', type=str)
    parser.add_argument('--final_path', type=str)
    args = parser.parse_args()
    
    denoised_model = DenoisedModel(args.denoiser_path, args.classifier_path)
    final_model = FinishedModel(denoised_model)
    torch.save(final_model, args.final_path)
