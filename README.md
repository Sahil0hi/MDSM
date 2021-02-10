# MDSM
Code for reproducing results in [Multiscale Denoising Score Matching](https://arxiv.org/abs/1910.07762)

## About MDSM
MDSM train a neural network Energy-Based Model Efficiently without sampling.

The resulting Energy function can be sampled with Annealed Langevin Dynamics.

Sampling process:
![sampling process](imgs/sampling.PNG|width=480)

CIFAR and CelebA samples:
![samples](imgs/samples.PNG)

## Requirements
Pytorch 
torchvision

## Usage
Train EBM on Fashion MNIST:
sh exps/fmnist_train.sh

Generate samples on Fashion MNIST (modify the time string in .sh file to that of your saved experiment before running):
sh exps/fmnist_sample_single.sh


Train EBM on CIFAR (takes about 24h on 2*2080Ti GPUs):
(requires ~8G of GPU memory in total, reduce batch size in the sh file if out of memory)
sh exps/cifar_train.sh

Generate samples for range of saved networks in a folder:
(modify time argument in sh file to that of you logging folder)
sh exps/cifar_sample_all.sh 

Generate more samples from one network (modify --log argument to folder name and --time argument to time string):
sh exps/cifar_sample_single.sh

## Pretrained models
download pretrained cifar model at (need to wrap model with DataParallel to work properly):
https://drive.google.com/open?id=18uH6UJJjjrdTX8qAf4YMNKFtBmW5o60k 
unpack to logs folder

after that, visualize samples from pretrain:
sh exps/cifar_visualize_pretrain.sh


## Other usage
Please write you custom script for inpainting experiment using the function Annealed_Langevin_E_mask in functions.sampling
