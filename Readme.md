# GraphEcho: Graph-Driven Unsupervised Domain Adaptation for Echocardiogram Video Segmentation
## Ver 0.3（PyTorch）

#### Project created by Jiewen Yang

This project is the pytorch implemention of Paper **"GraphEcho: Graph-Driven Unsupervised Domain Adaptation for Echocardiogram Video Segmentation"**. 
**[[paper](https://arxiv.org/abs/2309.11145)]**

Our experimental platform is configured with 1 RTX3090 (cuda>=11.0).  

Currently, this code is for public dataset CAMUS and EchoNet. 

The code that related to dataset **"CardiacUDA"** will be release after two weeks.


## Install


You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

Install Environment:
```
conda env create -f requirements.yaml
```

It is recommended to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## Data Preparation

This project provides the use case of echocardiogram video segmentation task;

The address index of the dataset can be found in the **train.py**, where you could do the parameters modification;

For different tasks, the composition of data sets have significant different, so there is no repetition in this file;



## 1. Download The *CAMUS* & *EchoNet* Dataset

The detail of CAMUS, please refer to: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html/.
The detail of EchoNet, please refer to: https://echonet.github.io/dynamic/.

Download the dataset.

The CAMUS dataset is composed as: /testing & /training
The EchoNet dataset is consist of: /Video, FileList.csv & VolumeTracings.csv

## Training

In this framework, after the parameters are configured in the file **train.py**, you only need to use the command:

```shell
python train.py
```

You are also able to start distributed training. 

**Note:** Please set the number of graphics cards you need and their id in parameter **"enable_GPUs_id"**.

###Code Reference 
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
https://github.com/chengchunhsu/EveryPixelMatters 