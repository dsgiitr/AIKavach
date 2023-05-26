from train_denoiser import *
import sys, getopt
import argparse

data=None
name = 'test'

parser = argparse.ArgumentParser(description='Denoiser Training')
# python3 train.py --epochs --dataset --input_dim --output_dim --model_name --model_path --no_of_channels --hidden_state_dim

# Training Objective
parser.add_argument('--epochs', type=int, help="Number of epochs")

# Dataset
parser.add_argument('--dataset', type=str, help='path to dataset of choice')
parser.add_argument('--in_nc', type=int, help='input dimensions')
parser.add_argument('--out_nc', type=int, help='output dimensions')

# Model type
parser.add_argument('--model_name', type=str, help="name of model")
parser.add_argument('--model_path', type=str, help="path to model")

# Setting

args = parser.parse_args()

de=denoiser(in_nc = args.in_nc,out_nc=args.out_nc,nc = [64,128,256,512], nb=4)
if args.model_path:
    de.ld(args.model_path)
de.train_drunet(args.epochs,args.dataset)
de.drunet.save(args.model_name)
# <<<<<<< dev
# de.drunet.save(args.model_name)
# =======
# de.drunet.save(args.model_name)
# >>>>>>> main
