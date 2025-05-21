# MDSM
Code for reproducing results in [Multiscale Denoising Score Matching](https://arxiv.org/abs/1910.07762)

## About MDSM
MDSM train a neural network Energy-Based Model Efficiently without sampling.

The resulting Energy function can be sampled with Annealed Langevin Dynamics.

CIFAR and CelebA samples:
![samples](imgs/samples.PNG)

## Requirements
 * PyTorch

 * torchvision

## Usage
Train EBM on Fashion MNIST:
```bash
sh exps/fmnist_train.sh
```

Generate samples on Fashion MNIST (modify the time string in .sh file to that of your saved experiment before running):
```bash
sh exps/fmnist_sample_single.sh
```

Train EBM on CIFAR (takes about 24h on 2 GPUs):
```bash
sh exps/cifar_train.sh
```

Generate samples for range of saved networks in a folder:
(modify time argument in sh file to that of you logging folder)
```bash
sh exps/cifar_sample_all.sh 
```

Generate more samples from one network (modify --log argument to folder name and --time argument to time string):
```bash
sh exps/cifar_sample_single.sh
```

## Pretrained models
Download pretrained cifar model at [here](https://drive.google.com/open?id=18uH6UJJjjrdTX8qAf4YMNKFtBmW5o60k ), and unpack to `logs` folder

Then, visualize samples from pretrain model with:
```bash
sh exps/cifar_visualize_pretrain.sh
```

## Citation
```bib
@article{li2019learning,
  title={Learning energy-based models in high-dimensional spaces with multi-scale denoising score matching},
  author={Li, Zengyi and Chen, Yubei and Sommer, Friedrich T},
  journal={arXiv preprint arXiv:1910.07762},
  year={2019}
}
```

