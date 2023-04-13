import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torch
from models.network_unet import UNetRes as net
from data.dataset_fdncnn import DatasetFDnCNN as FDnCNNdata

class denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.drunet = net(in_nc=4,
                   out_nc=3,
                   nc=[64, 128, 256, 512],
                   nb=4,
                   act_mode=  'R',
                   downsample_mode= 'strideconv',
                   upsample_mode= 'convtranspose',
                   bias= False)
        self.train_set=None
        self.train_size= None
        self.train_loader = None
    
    def create_dataloader(self,pth='trainsets/trainH'):
        """{'name': 'train_dataset', 'dataset_type': 'fdncnn', 'dataroot_H': 'trainsets/trainH', 
        'dataroot_L': None, 'H_size': 128, 'dataloader_shuffle': True, 'dataloader_num_workers': 8, 
        'dataloader_batch_size': 64, 'phase': 'train', 'scale': 1, 'n_channels': 3}"""
        self.batch_size=64
        self.train_set = FDnCNNdata(n_c=3,H_size=128, sigma=[0,75],sigma_test=25,dataroot_H=pth)
        self.train_size = int(math.ceil(len(self.train_set) / self.batch_size))
        self.train_loader = DataLoader(self.train_set,
                                          batch_size= self.batch_size,
                                          shuffle=True,
                                          num_workers= 8,
                                          drop_last=True,
                                          pin_memory=True)
    

    def initialise_drunet(self,pth='model_zoo/drunet_color.pth'):
        self.drunet.load_state_dict(torch.load(pth), strict=True)
        
    def train_drunet(self,epochs=1000):
        model=self.drunet
        for epoch in range(epochs):  # keep running
            print(epoch)
            for i, train_data in enumerate(self.train_loader):

                current_step += 1
                checkpoint_save=500

                # update learning rate
                model.update_learning_rate(current_step)

                # feed patch pairs
                model.feed_data(train_data)

                # optimize parameters
                model.optimize_parameters(current_step)

                # save model
                if current_step % checkpoint_save == 0:
                    model.save(current_step)

de=denoiser()
model=de.drunet
de.create_dataloader()
de.initialise_drunet()
de.train_drunet()
#print(model)
