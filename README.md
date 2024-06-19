<div align=center>
<h1> GraphEcho: Graph-Driven Unsupervised Domain Adaptation for Echocardiogram Video Segmentation </h1>
</div>
<div align=center>

<a src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square" href="https://arxiv.org/abs/2309.11145">
<img src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square">
</a>
   
<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>

<a src="https://img.shields.io/badge/%F0%9F%9A%80-XiaoweiXu's Github-blue.svg?style=flat-square" href="https://github.com/XiaoweiXu/CardiacUDA-dataset">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-Xiaowei Xu's Github-blue.svg?style=flat-square">
</a>

</div>


## :hammer: PostScript
&ensp; :smile: This project is the pytorch implemention of **[[paper](https://arxiv.org/abs/2309.11145)]**;

&ensp; :laughing: Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

&ensp; :blush: Currently, this code is avaliable for public dataset <u>CAMUS and EchoNet</u>;

&ensp; :smiley: For codes and accessment that related to dataset ***CardiacUDA***;

&ensp; &ensp; &ensp;    **:eyes:** The code is now available at:
&ensp; &ensp; &ensp;       ```
                            ..\datasets\cardiac_uda.py
                           ```

&ensp; :heart_eyes: For codes and accessment that related to dataset ***CardiacUDA***

&ensp; &ensp; &ensp;    **:eyes:** Please follw the link to access our dataset：


## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

2. Install Environment:
    ```
    conda env create -f requirements.yaml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

### *1. EchoNet & CAMUS*
 * This project provides the use case of echocardiogram video segmentation task;

 * The hyper parameters setting of the dataset can be found in the **train.py**, where you could do the parameters modification;

 * For different tasks, the composition of data sets have significant different, so there is no repetition in this file;


   #### *1.1. Download The **CAMUS**.*
   :speech_balloon: The detail of CAMUS, please refer to: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html/.

   1. Download & Unzip the dataset.

      The ***CAMUS dataset*** is composed as: /testing & /training.

   2. The source code of loading the CAMUS dataset exist in path :

      ```python
      ..\datasets\camus.py
      and modify the dataset path in
      ..\train_camus_echo.py
      ```
      New Version : We have updated the infos.npy in our new released code

   #### *1.2. Download The **EchoNet**.*

   :speech_balloon: The detail of EchoNet, please refer to: https://echonet.github.io/dynamic/.

   1. Download & Unzip the dataset.

      - The ***EchoNet*** dataset is consist of: /Video, FileList.csv & VolumeTracings.csv.

   2. The source code of loading the Echonet dataset exist in path :

      ```python
      ..\datasets\echo.py
      and modify the dataset path in
      ..\train_camus_echo.py
      ```

## *2. CardiacUDA*
 1.  Please access the dataset through : [XiaoweiXu's Github](https://github.com/XiaoweiXu/CardiacUDA-dataset)
 2.  Follw the instruction and download.
 3.  Finish dataset download and unzip the datasets.
 4.  Modify your code in both:
        ```python
        ..\datasets\cardiac_uda.py
        and modify the infos and dataset path in
        ..\train_cardiac_uda.py
        # The layer of the infos dict should be :
        # dict{
        #     center_name: {
        #                  file: {
        #                        views_images: {image_path},
        #                        views_labels: {label_path},}}}
        ```

## :feet: Training

1. In this framework, after the parameters are configured in the file **train_cardiac_uda.py** and **train_camus_echo.py**, you only need to use the command:

    ```shell
    python train_cardiac_uda.py
    ```
    And
    ```shell
    python train_camus_echo.py
    ```

2. You are also able to start distributed training. 

   - **Note:** Please set the number of graphics cards you need and their id in parameter **"enable_GPUs_id"**.

#


###### :rocket: Code Reference 
  - https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
  - https://github.com/chengchunhsu/EveryPixelMatters 

###### :rocket: Updates Ver 1.0（PyTorch）
###### :rocket: Project Created by Jiewen Yang : jyangcu@connect.ust.hk