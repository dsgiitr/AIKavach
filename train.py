from train_denoiser import *
import sys, getopt
import argparse

data=None
name = 'test'
opts, args = getopt.getopt(sys.argv[1:],"e:d:m:n:")
print(opts)
for opt, arg in opts:
    if opt =='-e':
        epoch = int(arg)
    if opt =='-d':
        data= arg
    if opt =='-m':
        model=arg    
    if opt =='-n':
        name = arg

de=denoiser()
de.ld(model)
de.train_drunet(epoch,data)
de.drunet.save(name)