# Code for Hat EBM

This repository will reproduce the main results from our paper

**Learning Probabilistic Models from Generator Latent Spaces with Hat EBM**<br/>Mitch Hill, Erik Nijkamp, Jonathan Mitchell, Bo Pang, Song-Chun Zhu<br/>https://arxiv.org/pdf/2210.16486.pdf<br/>NeurIPS 2022.

All experiments use Tensorflow 2. Both multi-GPU and TPU are supported.

## Setup

For CPU or GPU computing, run:

```pip install -r requirements.txt```

For TPU computing, run:

```pip install -r requirements_tpu.txt```

The TPU requirements are standard packages for TPU setup found [here](https://github.com/tensorflow/models/blob/master/official/requirements.txt). 

## Data Access

For CPU/GPU jobs, you will need to specify the path to the TF Records in the ```data_path``` variable in each config. In this case, you can set ```gs_path``` to ```None``` in the config file. 

For TPU jobs, you will need to specify the Google Storage Bucket name where the TF Record data is stored in the ```gs_path``` variable in each config. The setup assumes the data files are in a top-level folder in the bucket. In this case, you can set ```data_path``` to ```None``` in the config file.

Note that only TF Records for CIFAR-10 can be downloaded automatically, while CelebA and ImageNet 2012 will require a manual setup of the TF Records from the source files. Code for preparing TF Records from source data files can be found in ```prepare_tf_records.py```.

## Running a Training Job

To run a training experiment, you will need an executable file ```train_hat_ebm_joint.py``` (for retrofit and refinement training), ```train_hat_ebm_synth.py``` (for synthesis training), or ```train_ae.py```, and a config from the the ```configs_joint``` or ```config_synth``` folder. The execution command is

```python3 TRAIN_FILE CONFIG_FOLDER/CONFIG_FILE```

For example, to run a synthesis experiment on CIFAR-10, you can use the command

```python3 train_hat_ebm_synth.py configs_synth/cifar10_synth.py```

## Synthesis Experiments

Synthesis experiments will train an EBM and generator model in tandem to perform image synthesis. The folder ```configs_synth``` has files for CIFAR-10 at 32x32 resolution, Celeb-A at 64x64 resolution, and ImageNet 2012 at 128x128 resolution with a configs for a standard size net and a large size net with double the channels. At least 8 GPUs or a TPU-8 are recommended for running ImageNet experiments, and 32 GPUs or a TPU-32 are recommended for the large scale ImageNet experiment.

## Retrofit Experiments

The retrofit experiment in our paper first trained a deterministic autoencoder, then used the generator as part of the energy function of a Hat EBM. The file ```train_ae.py``` will train the deterministic autoencoder with the config file ```configs_joint/cifar10_ae.py```. The inference network is then discarded and the generator weight path should be specified in the config file ```cifar10_retrofit.py```, which will train the Hat EBM.

## Refinement Experiment

This experiment will take a generator trained from an SN-GAN and train a Hat EBM with the generator to learn to refine the generator samples in the latent space. The pretrained SN-GAN generator can be found in the files ```pretrained_nets/cifar10_refine/gen.ckpt``` and ```pretrained_nets/celeb_a_refine/gen.ckpt``` with the pretrained networks released with this repo.

## FID Evaluation

The ```fid``` folder contains the executable files to perform FID evaluation and config files that will reproduce the scores from our paper for each of the pretrained nets. The file ```fid_save_ims.py``` can be used to visualize samples from a trained network and to save samples in numpy format that can be used for FID evaluation with the original TF1 FID code. An example command for running this file is

```python3 fid/fid_save_ims.py --config_name fid/fid_config.py```

The file ```fid_orig.py``` contains the original FID code that is adapted to read the numpy files saved by ```fid_save_ims.py```. This can be run with the command

```python3 fid/fid_orig.py --path1 /PATH/TO/images1.npy --path2 /PATH/TO/images2.npy```

## Pretrained Nets

The pretrained networks for each experiment can be found in the ```releases``` section of the repo.

## TPU Support Acknowledgement

This code was developed with Cloud TPUs from Google's TPU Research Cloud (TRC).

## Contact

Please direct any inquiries to Mitch Hill at ```point0bar1@gmail.com```.
