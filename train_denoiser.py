import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
# import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from extra import utils_logger
from extra import utils_image as util
from extra import utils_option as option
from extra.utils_dist import get_dist_info, init_dist

# from data.select_dataset import define_Dataset
from models.network_unet import UNetRes as net
from models.model_plain import ModelPlain as Plain
from data.dataset_fdncnn import DatasetFDnCNN as FDnCNNdata


'''
# --------------------------------------------
# training code for DRUNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
'''

class denoiser():
    def __init__(self,json_path='options/train_drunet.json') -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
        parser.add_argument('--launcher', default='pytorch', help='job launcher')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--dist', default=False)
        parser.add_argument('-d', default=0)
        parser.add_argument('-e', type=int,default=False)
        parser.add_argument('-m',  default=0)
        parser.add_argument('-n')
        opt = option.parse(parser.parse_args().opt, is_train=True)
        opt['dist'] = parser.parse_args().dist

        # ----------------------------------------
        # distributed settings
        # ----------------------------------------
        if opt['dist']:
            init_dist('pytorch')
        opt['rank'], opt['world_size'] = get_dist_info()

        if opt['rank'] == 0:
            util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

        # ----------------------------------------
        # update opt
        # ----------------------------------------
        # -->-->-->-->-->-->-->-->-->-->-->-->-->-
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        opt['path']['pretrained_netG'] = init_path_G
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
        self.current_step = max(init_iter_G, init_iter_optimizerG)

        border = opt['scale']
        # --<--<--<--<--<--<--<--<--<--<--<--<--<-

        # ----------------------------------------
        # save opt to  a '../option.json' file
        # ----------------------------------------
        if opt['rank'] == 0:
            option.save(opt)

        # ----------------------------------------
        # return None for missing key
        # ----------------------------------------
        self.opt = option.dict_to_nonedict(opt)

        self.drunet =Plain(self.opt)
        self.train_set=None
        self.train_size= None
        self.train_loader = None

    def ld(self,pth):
        self.drunet.load(pth)

    def train_drunet(self,epochs=1000,pth=None):

        # Step--1 (prepare opt)
    
        # ----------------------------------------
        # seed
        # ----------------------------------------
        # print(opt)
        seed = self.opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        print('Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        '''
        # ----------------------------------------
        # Step--2 (creat dataloader)
        # ----------------------------------------
        '''

        # ----------------------------------------
        # 1) create_dataset
        # 2) creat_dataloader for train and test
        # ----------------------------------------
        if pth is not None:
            train_set = FDnCNNdata(dataroot_H=pth)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            print(train_size)
            if self.opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                        shuffle=False,
                                        num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                        drop_last=True,
                                        pin_memory=True,
                                        sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=dataset_opt['dataloader_shuffle'],
                                        num_workers=dataset_opt['dataloader_num_workers'],
                                        drop_last=True,
                                        pin_memory=True)

        else:
            for phase, dataset_opt in self.opt['datasets'].items():
                if phase == 'train':
                    train_set = FDnCNNdata()
                    train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
                    if self.opt['dist']:
                        train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                        train_loader = DataLoader(train_set,
                                                batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                                shuffle=False,
                                                num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                                drop_last=True,
                                                pin_memory=True,
                                                sampler=train_sampler)
                    else:
                        train_loader = DataLoader(train_set,
                                                batch_size=dataset_opt['dataloader_batch_size'],
                                                shuffle=dataset_opt['dataloader_shuffle'],
                                                num_workers=dataset_opt['dataloader_num_workers'],
                                                drop_last=True,
                                                pin_memory=True)

                elif phase == 'test':
                    test_set = FDnCNNdata()
                    test_loader = DataLoader(test_set, batch_size=1,
                                            shuffle=False, num_workers=1,
                                            drop_last=False, pin_memory=True)
                else:
                    raise NotImplementedError("Phase [%s] is not recognized." % phase)

        '''
        # ----------------------------------------
        # Step--3 (initialize model)
        # ----------------------------------------
        '''

        model =self.drunet
        model.init_train()
    

        '''
        # ----------------------------------------
        # Step--4 (main training)
        # ----------------------------------------
        '''

        for epoch in range(epochs):  # keep running
            print(epoch)
            # if opt['dist']:
            #     train_sampler.set_epoch(epoch)
            for i, train_data in enumerate(train_loader):

                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

                self.current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                model.update_learning_rate(self.current_step)

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                model.optimize_parameters(self.current_step)

                # -------------------------------
                # 4) training information
                # -------------------------------
                if self.current_step % self.opt['train']['checkpoint_print'] == 0 and self.opt['rank'] == 0:
                    # logs = model.current_log()  # such as loss
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, self.current_step, model.current_learning_rate())
                    print(message)
                    


                # -------------------------------
                # 5) save model
                # -------------------------------
                if self.current_step % self.opt['train']['checkpoint_save'] == 0 and self.opt['rank'] == 0:
                    
                    model.save(self.current_step)

                # -------------------------------
                # 6) testing
                # -------------------------------
                if self.current_step % self.opt['train']['checkpoint_test'] == 0 and self.opt['rank'] == 0:

                    avg_psnr = 0.0
                    idx = 0

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        img_dir = os.path.join(self.opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(test_data)
                        model.test()

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, self.current_step))
                        util.imsave(E_img, save_img_path)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                        

                        avg_psnr += current_psnr

                    avg_psnr = avg_psnr / idx

                    print(avg_psnr)
        self.drunet =model

