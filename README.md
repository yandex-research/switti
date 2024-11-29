# Switti: Designing Scale-Wise Transformers for Text-to-Image Synthesis

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

# Model weights
Our final checkpoints can be downloaded via following links:
|   model    | reso | weights link |
|:----------:|:-----:|:-----:|
| RQ-VAE (FT)   | 512x512 | [vae_ft_checkpoint.pt](https://www.dropbox.com/scl/fi/i6py66o26phjqz4xmkgd2/vae_ft_checkpoint.pt?rlkey=onp0zxarht7z5a3c8g3ojmewz&st=bsw9z6fu&dl=1) |
|  Switti (AR)   | 512x512 | [switti_ar.pt](???) |
|  Switti   | 512x512 | [model_state_dict.pt](https://www.dropbox.com/scl/fi/99a6v8to1ib0bdlgvoi0r/model_state_dict.pt?rlkey=vs6ofh3w1bg4m31y4h3hj0opn&st=pyihw6an&dl=0) |

Alternatively, it is possible to run Switti with non-finetuned RQ-VAE from VAR, that can be downloaded via this [link](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth)

# Training

## Data
We provide the training example on the [COCO](https://cocodataset.org/#download) dataset one needs to prepare `--data_path` in the following format:
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

## Training script
A minimal example of training on several GPUs using FSDP can be found at `sh_scripts/run_example.sh`.

During training, intermediate checkpoints and tensorboard logs will be saved to the `local_output` folder.

Set `--use_ar=true` to train an AutoRegressive version of Switti

# Inference
After you download the weights of our fine-tuned RQ-VAE and Switti generator, you can experiment with inference using various sampling parameters, following `inference_example.ipynb`
