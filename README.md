
<div align="center">

# Walk Video Classification

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->

<!--  
Conference   
-->
</div>

## Description  

📓 This project made with the PyTorch, PyTorch Lightning, PyTorch Video.

This project implements the task of classifying different medical diseases.

The current phase performs a dichotomous classification task for four different disorders. classification of ASD and non-ASD.

The whole procedure is divided into two steps:  

1. using the detection method to extract the character-centered region and save it as a video.
2. put the 1. processed video into 3D CNN network for training.

Detailed comments are written for most of the methods and classes.
Have a nice code. 😄

## Folder structure  

``` bash
Walk_Video_PyTorch
|-- imgs
|   imgs for markdown.
|-- logs
|   output logs and saved model .ckpt file location.
`-- project
    |-- dataloader
    |   pytorch lightning data module based dataloader, to prepare the train/val/test dataloader for different stage.
    |-- models
    |   pytorch lightning modeul based model, where used for train and val.
    |-- prepare_video
    |   one stage prepare video code location
    |-- tests
    |   `-- logs
    |   test logs location and code
    `-- utils
        tools for the code.
```

## How to run

1. install dependencies

``` bash
# clone project   
git clone https://github.com/ChenKaiXuSan/Walk_Video_PyTorch.git

# install project   
cd Walk_Video_PyTorch/ 
pip install -e .   
pip install -r requirements.txt
```

2. navigate to any file and run it.

```bash
# module folder
cd Walk_Video_PyTorch/

# run module 
python project/main.py [option] > logs/output_log/xxx.log 
```

### one stage  

In this stage, the interface provided by [detector2](https://detectron2.readthedocs.io/en/latest/index.html) is used for video pre-processing, in order to extract the area centered on the person and save a series of frames as video with a uniform FPS = 30.

The implementation of the prepare_video.py file,  

``` python  
usage: prepare_video.py [-h] [--img_size IMG_SIZE] [--num_workers NUM_WORKERS] [--data_path DATA_PATH] [--split_pad_data_path SPLIT_PAD_DATA_PATH] [--split_data_path SPLIT_DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE
  --num_workers NUM_WORKERS
                        dataloader for load video
  --data_path DATA_PATH
                        meta dataset path
  --split_pad_data_path SPLIT_PAD_DATA_PATH
                        split and pad dataset with detection method.
  --split_data_path SPLIT_DATA_PATH
                        split dataset with detection method.

```

for example,  

``` python  
cd Walk_Video_PyTorch/project/prepare_video/

python prepare_video.py --img_size 512 --data_path [meta dataset path] --split_pad_data_path [split and pad dataset path] --split_data_path [split dataset path] > ./split_log.log &

```

⚠️ You need to replace the content in [ ] with your own actual path.

### two stage  

In this stage, the pre-processed video needs to be read and fed to the 3d cnn network for training.

the implementation of the main.py file.  

``` python

usage: main.py [-h] [--model {resnet,csn,x3d}] [--img_size IMG_SIZE] [--version VERSION] [--model_class_num MODEL_CLASS_NUM] [--model_depth {50,101,152}] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--clip_duration CLIP_DURATION]
               [--uniform_temporal_subsample_num UNIFORM_TEMPORAL_SUBSAMPLE_NUM] [--gpu_num {0,1}] [--lr LR] [--beta1 BETA1] [--beta2 BETA2] [--data_path DATA_PATH] [--split_data_path SPLIT_DATA_PATH] [--split_pad_data_path SPLIT_PAD_DATA_PATH] [--log_path LOG_PATH]
               [--pretrained_model PRETRAINED_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --model {resnet,csn,x3d}
  --img_size IMG_SIZE
  --version VERSION     the version of logger, such data
  --model_class_num MODEL_CLASS_NUM
                        the class num of model
  --model_depth {50,101,152}
                        the depth of used model
  --max_epochs MAX_EPOCHS
                        numer of epochs of training
  --batch_size BATCH_SIZE
                        batch size for the dataloader
  --num_workers NUM_WORKERS
                        dataloader for load video
  --clip_duration CLIP_DURATION
                        clip duration for the video
  --uniform_temporal_subsample_num UNIFORM_TEMPORAL_SUBSAMPLE_NUM
                        num frame from the clip duration
  --gpu_num {0,1}       the gpu number whicht to train
  --lr LR               learning rate for optimizer
  --beta1 BETA1
  --beta2 BETA2
  --data_path DATA_PATH
                        meta dataset path
  --split_data_path SPLIT_DATA_PATH
                        split dataset path
  --split_pad_data_path SPLIT_PAD_DATA_PATH
                        split and pad dataset with detection method.
  --log_path LOG_PATH   the lightning logs saved path
  --pretrained_model PRETRAINED_MODEL
                        if use the pretrained model for training.
```

for example,  

``` python  

python project/main.py --version [the version for train] --model resnet --model_depth 50 --img_size [img size] --batch_size [batch size] --clip_duration [the clip duration] --uniform_temporal_subsample_num [how many frames for train] --num_workers 16 > logs/output_logs/[version].log 2>&1 & 

```

⚠️ You need to replace the content in [ ] with your own actual path.

# Experimental setup

## Dataset

Due to the limitation of data, we divide 80% of the data into the training set and set the remaining 20% as the validation and test data sets.

The detail number of the split dataset are given in next table.

| the number of video | ASD | ASD_not |
| ------------------- | --- | ------- |
| train               | 923 | 815     |
| val                 | 123 | 96      |

The detail number of different disease and the number of every disease object are given in next talbe.

| disease | number of video | object for train | object for val |
| ------- | --------------- | ---------------- | -------------- |
| ASD     | 1046            | 48               | 6              |
| DHS     | 587             | 14               | 2              |
| HipOA   | 64              | 3                | null           |
| LCS     | 260             | 7                | 2              |

## Data augmentation  

Because the detection method is used to first extract the features of the characters into squares, there is no need to consider the border inconsistency when convolving.
We use both spatial and temporal jittering for augmentation.

Each clip is then generated by randomly cropping windows of size 224x224.
We train and evaluate models with clips of different frames (T = 16, 32, 64) by skipping every other frame (all videos are pre-processed to 30fps).

At the [data_loader.py](project/dataloader/data_loader.py) we defined the transform function.

``` python
self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),

                            Div255(),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
```

## Network Structure

So far, we have used the 3D Resnet structure, which are given in the next figure.
![network](imgs/network.png)

## Docker  

We recommend using docker to build the training environment.

1. pull the official docker image, where release in the [pytorchlightning/pytorch_lightning](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)

``` bash  
docker pull pytorchlightning/pytorch_lightning
```

2. create container.

``` bach  
docker run -itd -v $(pwd)/path:/path --gpus all --name container_name --shm-size 32g --ipc="host" <images:latest> bash 

```

3. enter the container and run the code.

``` bash  
docker exec -it container_name bash
```

## About the lib  

stop building wheels 😄

### PyTorch Lightning  

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.

### PyTorch Video  

[link](https://pytorchvideo.org/)
A deep learning library for video understanding research.

### detectron2

[Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.

### Torch Metrics

[TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) is a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.
