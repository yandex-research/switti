# Switti: Designing Scale-Wise Transformers for Text-to-Image Synthesis
<a href='https://arxiv.org/'><img src='https://img.shields.io/badge/ArXiv-Paper-red'></a> &nbsp; 
<a href='https://yandex-research.github.io/switti/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
<a href="https://huggingface.co/spaces/">
	    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20-Demo-orange' />
</a>&nbsp;

We present Switti, a scale-wise transformer for text-to-image generation that outperforms existing T2I AR models and competes with state-of-the-art T2I diffusion models while being faster than distilled diffusion models.

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

This repo contains code required to reproduce training of our models (Switti and Switti (AR)) and a notebook with an inference example.

# Setup

First, create a new environment and install Pytorch using conda:
```bash
conda create -n switti python=3.11
conda activate switti
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Next, install other libraries required for training and inference via pip
```bash
pip install -r requirements.txt
```

## [Optional] Install Apex
Apex is not essential to run our code, however it can accelerate both training and inference to some extent.

To install apex from source:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Training

## Data
We provide the training example on the [COCO](https://cocodataset.org/#download) dataset, that needs to be prepared in the following format:
```
/path/to/data/:
    train2014/:
        COCO_train2014_000000000009.jpg
        COCO_train2014_000000000025.jpg
        ...
    train2014.csv
```
Where `/path/to/data/train2014.csv` is a table with the following header:
```csv
,file_name,caption
0,COCO_train2014_000000057870.jpg,A restaurant has modern wooden tables and chairs.
```
Path to the folder containing the prepared COCO dataset is used as `--data_path` argument in a training script.

## Training script
A minimal example of training on several GPUs using FSDP can be found at `sh_scripts/train_example.sh`.

During training, intermediate checkpoints and tensorboard logs will be saved to the `local_output` folder.

Set `--use_ar=true` to train an AutoRegressive version of Switti

# Inference
You can experiment with Switti inference using various sampling parameters via [HuggingFace demo](https://) or a Jupyter notebook [inference_example.ipynb](inference_example.ipynb).

# Citation
If you make use of our work, please cite our paper:
```
```