import torch
import argparse

parser = argparse.ArgumentParser(description='The main script for DSRS')
# parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'imagenet', 'tinyimagenet'])
# parser.add_argument('--original_rad_dir', type=str, default='data/orig-radius')
parser.add_argument('--path_to_pretrained_model', type=str, default='None')
parser.add_argument('--input_dims', type=list, default=[3, 32, 32], help="expected dimensions of the input in the order channel, height, width")

args = parser.parse_args()

raw_model = torch.load(args.path_to_pretrained_model)
distributions
## get trained denoiser
## get optimal 

