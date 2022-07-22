# README

---

## Description

---

This repository provides a script for training a 2D U-Net of liver segmentation in CT volumes.  
It is designed to be run on the dl1.24xlarge, p3dn.24xlarge and p4d.24xlarge EC2 instances on AWS.  
Monitoring metrics are logged to evaluate the performance of Gaudi cards and Nvidia GPUs.

## Setup

---

### Clone the project repository

```bash
cd
git clone https://github.com/Adrien3198/HabanaBenchmark.git LiverSeg
```

### Environment

- Ubuntu 20.04
- Python 3.8
- Tensorflow 2.8

To launch a training on Habana dl1.24xlarge EC2 instance, use the **Deep Learning AMI Habana TensorFlow 2.8.0 SynapseAI 1.3.0 (Ubuntu 20.04)**

For Nvidia GPUs based EC2 instances, use the **Deep Learning AMI GPU TensorFlow 2.8.0 (Ubuntu 20.04)**

Set PYTHON environment variable to `/usr/bin/python3.8`

Then, install required package with the setup script :

```bash
bash LiverSeg/env/install.sh
```

### Download the dataset

Download the dataset on Kaggle at:

- [Liver Tumor Segmentation pt1](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
- [Liver Tumor Segmentation pt2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)

Unzip all files in a single directory: `~/DATA/original_data`

### Preprocess and train-test split the dataset

Go to the project scripts directory:

```bash
cd ~/LiverSeg/scripts
```

Run the preprocessing and train-test splitting script:

```bash
bash preprocessing_train_test.sh
```

## Training

---

To launch a training task on GPUs, run the command:

```bash
horovodrun -np <number_of_workers> $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance <instance_type {dl1n, p4d, p3dn}> -bs <batch_size> -e <number_of_epochs> -l <tensorboard_log_dir>

```

Example:

- Running on 8 GPUs on p4d instance with batch size 32 and 100 epochs:

```bash
horovodrun -np 8 $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance p4d -bs 32 -e 100 -l tensorboard_logs
```

- Running on 8 Gaudi cards on dl1 with batch size 32 and 100 epochs:

```bash
horovodrun -np 8 $PYTHON ~/LiverSeg/scripts/train.py -i ~/DATA/training_data -instance dl1 -bs 32 -e 100 -l tensorboard_logs --gaudi

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
