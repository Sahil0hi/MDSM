import numpy as np    
import matplotlib.pyplot as plt
import pickle
import yaml
import argparse
import os
#import h5py as h5
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv





from models.ResNet import Res12_Quadratic, Res18_Quadratic, Res34_Quadratic
# from saveModels import saveHelper
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# To load and process the data.

def preload(path, start_sub = 0, num_sub_per_type = 2, acc = 4.0, num_sl = 10,Cartesian = False, \
            contrast_type = 3):
    #acc: acceleration 
    #num_sl: # of slices per subject
    #Cartesian = True(for Cartesian sampling),Cartesian = False(for Poisson sampling)
    #constrast_type: 0: FLAIR, 1:T1Post, 2:T1, 3:T2
    
    subdirs = sorted(os.listdir(path))
    train_ksp, train_csm, labels = None, None, None
    subdir = subdirs[contrast_type]
    fnames = [filename for filename in sorted(os.listdir(path+subdir)) if filename.endswith('.pickle')]
    print(subdir, '- loading', num_sub_per_type, 'of', len(fnames), 'subjects')
        
    subpath = os.path.join(path, subdir)
    train_fnames = fnames[start_sub:start_sub+num_sub_per_type]
        
    for j, train_fname in enumerate(train_fnames):
        with open(os.path.join(subpath, train_fname), 'rb') as f:
            ksp, csm = pickle.load(f)
            ksp, csm = ksp[:num_sl], csm[:num_sl]
            if j==0:
                train_ksp = torch.tensor(ksp)
                train_csm = torch.tensor(csm)
                labels = torch.ones(ksp.shape[0],)*contrast_type
            else:
                train_ksp = torch.cat((train_ksp, torch.tensor(ksp)))
                train_csm = torch.cat((train_csm, torch.tensor(csm)))
                labels = torch.cat((labels, torch.ones(ksp.shape[0],)*contrast_type))
            print('ksp:', ksp.shape, '\tcsm:', csm.shape)
        
    # print('ksp:', train_ksp.shape, '\ncsm:', train_csm.shape, '\nlabels:', labels.shape,)
    
    if acc == 0:
        mask = torch.ones_like(train_ksp)
    elif acc != None:
        if acc==4 and Cartesian==True:
            mask_filename ='acc4_c.npy'
            print("Acceleration = 4 and Cartesian sampling")
        if acc==2 and Cartesian==True:
            mask_filename ='acc2_c.npy'
            print("Acceleration = 2 and 1D sampling")
        elif acc==4 and Cartesian==False:
            mask_filename ='poisson_mask_2d_acc4.0_320by320.npy'
            print("Acceleration = 4 and Poisson sampling")
        if acc==6:
            mask_filename ='acc_6_t.npy'
            print("Acceleration = 6 and Poisson sampling")
        # mask = np.load(mask_filename).astype(np.complex64)  
        # mask = torch.tensor(np.tile(mask, [train_ksp.shape[0],train_ksp.shape[1],1,1]))
        mask = None
    else:
        mask = None
    
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    print(f"Loaded dataset of {train_ksp.shape[0]} slices\n")
    
    return train_ksp, train_csm, mask, labels.long(), labels_key



def preprocess(ksp, csm, mask):    
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    us_ksp = ksp #* mask
    
    return org, us_ksp.type(torch.complex64), csm.type(torch.complex64), mask

class DataGenBrain(Dataset):
    def __init__(self,  start_sub=0, num_sub=2, device=None, acc=4.0, mask_type=False, data_path='/sfs/ceph/standard/CBIG-Standard-ECE/aniket/FastMRI brain data/',contrast=3):
        self.path = data_path
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        self.acc = acc
        self.type=mask_type
        self.contrast = contrast
        self.ksp, self.csm, self.msk, self.labels, self.labels_key = preload(self.path, \
                                                                             self.start_sub, \
                                                                             self.num_sub, \
                                                                             acc = self.acc, \
                                                                             num_sl = 10, \
                                                                             Cartesian = self.type, \
                                                                             contrast_type = self.contrast)
        self.org, self.us_ksp, self.csm, self.msk = preprocess(self.ksp, self.csm, self.msk)


        
    def __len__(self):
        return self.org.size()[0]
   
        
    def __getitem__(self, i):
        return self.org[i:i+1].to(self.device)
        # return self.org[i:i+1].to(self.device), self.us_ksp[i:i+1].to(self.device), \
        #        self.csm[i:i+1].to(self.device), self.msk[i:i+1].to(self.device), \
        #        self.labels[i:i+1].to(self.device)
    
    def get_noisy(self, i, noise_eps=0.):
        us_ksp = self.us_ksp[i:i+1] 
        msk = self.msk[i:i+1]
        scale = 1/torch.sqrt(torch.tensor(2.))
        us_ksp = us_ksp + msk*(torch.randn(us_ksp.shape)+1j*torch.randn(us_ksp.shape))*scale*noise_eps
        
        return self.org[i:i+1].to(self.device)
        # return self.org[i:i+1].to(self.device), us_ksp.to(self.device), \
        #        self.csm[i:i+1].to(self.device), msk.to(self.device), \
        #        self.labels[i:i+1].to(self.device)

acc=4
rd=DataGenBrain(start_sub=1, num_sub=104, device=device, acc=acc,contrast=2) # num_sub used to be 64

EPOCHS_TRAINED = 0

fname = "experiment6"
loss_file_name = f'{fname}/loss.csv'
model_name = f'{fname}/models/netE_{EPOCHS_TRAINED}.pt'
n_epochs = 20000
tqdm_epoch = np.arange(EPOCHS_TRAINED + 1, n_epochs)
t_loss= []

netE =  Res18_Quadratic(2, 128, 320,normalize=False,AF=nn.ELU())
if os.path.exists(model_name):
    print('Loading checkpoint')
    netE.load_state_dict(torch.load(model_name, weights_only=False))
else:
    print('Starting from a fresh model')
    
netE = netE.to(device)

params = {'lr':1e-6,'betas':(0.9,0.95)} # used to be 5e-5
optimizerE = torch.optim.Adam(netE.parameters(),**params)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE,20000,eta_min=1e-8,last_epoch=-1) # used to be 1e-6

print('Dataset size', len(rd))

desired_batch_size = 128
batch_size = 32
parts = desired_batch_size // batch_size

data_loader = DataLoader(dataset=rd, batch_size=batch_size, shuffle=True)

sigmas_np = np.logspace(np.log10(0.002), np.log10(5), num=desired_batch_size)
# sigmas_np = np.logspace(0.05, 5, desired_batch_size)
# sigmas_np = np.linspace(0.05, 3, desired_batch_size)
sigmas = torch.Tensor(sigmas_np).view((desired_batch_size,1,1,1)).to(device)

# alphas_np = np.linspace(0.0001, 1 - 0.0001, desired_batch_size)
# alphas = torch.Tensor(alphas_np).view((desired_batch_size,1,1,1)).to(device)

sigma0 = 0.1
sigma02 = sigma0**2
lambda2 = 1e-5 # coefficient multiplied by pair loss

def augment_batch(batch, flip_type):
    # angle = float(np.random.choice([0, 90, 180, 270]))
    # org_rotated = torch.stack([TF.rotate(img, angle) for img in batch])
    # return org_rotated

    if flip_type == "horizontal":
        return TF.hflip(batch)
    else:
        return batch

for epoch in tqdm_epoch:
    
    avg_loss = torch.zeros(1, dtype=torch.double)
    # avg_sigma_loss = torch.zeros(1, dtype=torch.double)
    # avg_alpha_loss = torch.zeros(1, dtype=torch.double)
    num_items = 0
    optimizerE.zero_grad()

    transform = random.choice(["horizontal", "none"])

    for i, org in enumerate(data_loader):

        org = torch.squeeze(org, 1)
        org = org.to(device)
        org = torch.cat((org.real, org.imag), 1).float()
        org = org.to(device)

        org = augment_batch(org, transform)
        i %= parts
        
        if(org.shape[0]==batch_size): #eliminate small batches, which is a problem for linspace
            sigmas_part = sigmas[i*batch_size:(i+1)*batch_size]
            noise = sigmas_part * torch.randn_like(org)
            
            org_noisy = org + noise
            org_noisy = org_noisy.requires_grad_()
            E = netE(org_noisy).sum()
           
            grad_x = torch.autograd.grad(E, org_noisy, create_graph=True)[0]
            # org_noisy = org_noisy.detach()
            org_noisy.detach()
            
            sigma_loss = ((((org - org_noisy)/sigmas_part/sigma02+grad_x/sigmas_part)**2)/batch_size).sum() / float(parts) #+ lambda2 * (netE(org).sum() - netE(alpha * org).sum())
            sigma_loss.backward()
            
            # alphas_part = alphas[i::parts]
            
            # org_alpha = org * (-alphas_part) # amount of resolution we're removing from org
            # org_scaled = org + org_alpha # scaled img (less bright)
            # org_scaled = org_scaled.requires_grad_()
            # E = netE(org_scaled).sum()

            # grad_x = torch.autograd.grad(E, org_scaled, create_graph=True)[0]
            # org_alpha = org_alpha.detach()

            # alpha_loss = lambda2 * (((grad_x / (1 - alphas_part) - org_alpha / (1 - alphas_part) / sigma02)**2) / batch_size).sum() / float(parts) # essentially, we're trying to get the negative gradient to match the direction the img must go to get to the right scale
           
            # alpha_loss.backward()

            if i == parts - 1:
                optimizerE.step()
                optimizerE.zero_grad()
                transform = random.choice(["horizontal", "none"])

            with torch.no_grad():
                avg_loss += sigma_loss.item() * batch_size
                # avg_loss += (sigma_loss + alpha_loss).item() * batch_size
                # avg_sigma_loss += sigma_loss.item() * batch_size
                # avg_alpha_loss += alpha_loss.item() * batch_size    
                num_items += batch_size

    epoch_loss = (avg_loss/num_items).item()
    # epoch_sigma_loss = (avg_sigma_loss/num_items).item()
    # epoch_alpha_loss = (avg_alpha_loss/num_items).item()
    # print(epoch_loss, epoch_sigma_loss, epoch_alpha_loss)
    print(epoch_loss)
        
    t_loss.append(epoch_loss)
    scheduler.step()
        
    if(np.mod(epoch,10)==0):
        print("ep ",epoch, " energynet:", epoch_loss)
        #plt.imshow(score[0,0].abs().detach().cpu())
        #plt.show()
        with open(loss_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch.item(), epoch_loss]) #, epoch_sigma_loss, epoch_alpha_loss])
    if (np.mod(epoch,100)==0):
        torch.save(netE.state_dict(), f'{fname}/models/netE_{epoch}.pt')

plt.plot(torch.log(torch.tensor(t_loss)))