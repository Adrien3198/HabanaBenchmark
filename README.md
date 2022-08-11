# README

---

## Description

---

This repository provides a script for training a 2D U-Net of liver segmentation in CT volumes. 
It is designed to run on the dl1.24xlarge, p3dn.24xlarge and p4d.24xlarge EC2 instances on AWS.
Monitoring metrics are logged to evaluate the performance of Gaudi cards and Nvidia GPUs.


## Setup

To help developers find more information to get started with Habana AI processors, we suggest that you check the Habana developer site https://developer.habana.ai/
___
### Clone the project repository

```bash
cd ~
git clone https://github.com/Adrien3198/HabanaBenchmark.git LiverSeg
```

### Environment

- Ubuntu 20.04
- Python 3.8
- Tensorflow 2.8

To launch a training on Habana dl1.24xlarge EC2 instance, use the **Deep Learning AMI Habana TensorFlow 2.8.0 SynapseAI 1.4.0 (Ubuntu 20.04)**.

Learn more about the Habana setup information with the setup/install guide: https://docs.habana.ai/en/latest/Installation_Guide/index.html

For Nvidia GPUs based EC2 instances, use the **Deep Learning AMI GPU TensorFlow 2.8.0 (Ubuntu 20.04)**.

Set PYTHON environment variable to `/usr/bin/python3.8` with adding the following line at the end of the `~/.bashrc` file:  
```txt
PYTHON=/usr/bin/python3.8
```
And then run :  
```bash
source ~/.bashrc
```

For all instance types, you can install required packages described in `LiverSeg/env/requirements.txt` file with pip :

```bash
$PYTHON -m pip install -r ~/LiverSeg/env/requirements.txt
```

For dl1 instances, the habana-horovod custom package to use Horovod with Gaudi cards is already installed.

However, for GPU-based instances (p4d.24xlarge, p3dn.24xlarge), you have to install Horovod for TeansorFlow to run on multiple cards.
```bash
HOROVOD_WITH_TENSORFLOW=1 $PYTHON -m pip install -r horovod[tensorflow]
```

### Download the dataset

Download the dataset on Kaggle at:

- [Liver Tumor Segmentation pt1](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
- [Liver Tumor Segmentation pt2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)

Unzip all files in a single directory: `~/DATA/original_data` like :  

original_data/  
  |  volume-0.nii  
  |  segmentation-0.nii  
  |  volume-1.nii  
  |  segmentation-1.nii  
  |  ...

### Preprocess the dataset

Go to the project scripts directory:

```bash
cd ~/LiverSeg/scripts
```

Run these commands :

```bash
$PYTHON preprocess.py -i ~/DATA/original_data -t ~/DATA/preprocessed_data -f
```
```bash
$PYTHON create_train_test_dir.py -i ~/DATA/preprocessed_data -t ~/DATA/trainin
g_data -f
```

## Training

---

To launch a training task, run the command:

```bash
horovodrun -np <number_of_workers> $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance <instance_type {dl1n, p4d, p3dn}> -bs <batch_size> -e <number_of_epochs> -l <tensorboard_log_dir>

```

The instance name dl1 allow computaion on Gaudi. p4d and p3dn allow computation on GPU.

Example:

- Running on 8 GPUs on p4d instance with batch size 32 and 100 epochs:

```bash
horovodrun -np 8 $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance p4d -bs 32 -e 100 -l tensorboard_logs
```

- Running on 8 Gaudi cards on dl1 with batch size 32 and 100 epochs:

```bash
horovodrun -np 8 $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance dl1 -bs 32 -e 100 -l tensorboard_logs
```

To enable mixed_precision, add the `--mixed_precision` flag in the command for Gaudi and GPUs

Example:

- Running on 8 Gaudi cards with batch size 32 and 100 epochs and bfloat16:

```bash
horovodrun -np 8 $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance dl1 -bs 32 -e 100 -l tensorboard_logs --mixed_precision
```

During training, some performance logs are produced in the directory `performance_logs` where there are specific directories for each session. each node produces its own logs in a file of its own.  
Example:

```txt
2022-07-22 07:15:21,415 - INFO : Step 0 to 20 : 16.82 seconds, 38.05 examples/sec
```
