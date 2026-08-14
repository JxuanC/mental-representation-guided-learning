# Human-like Cognitive Generalization for Large Models via Brain-in-the-loop Supervision
Official Implementation in PyTorch

## Environment setup
Create a conda environment and install the packages necessary to run the code.
```
conda create -n brainintheloop python=3.8.13 -y
conda activate brainintheloop
pip install -r requirements.txt
```

## Reproducing the paper Figures
`Fig*.ipynb`: Jupyter notebooks for reproducing the paper figures from source data

## Download data and checkpoints
1. Download [DIR dataset](https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577) (Kamitani Lab). Please download them from FigShare and put them in this repository as shown below.
```
┣ 📂 dataset
┃   ┣ 📂 fMRI
┃   ┃   ┣ 📂 DeepImageReconstruction
┃   ┃   ┃   ┗ sub-01_perceptionNaturalImageTraining_VC_v2.h5
┃   ┃   ┃   ┗ sub-02_perceptionNaturalImageTraining_VC_v2.h5
┃   ┃   ┃   ┗ sub-03_perceptionNaturalImageTraining_VC_v2.h5
┃   ┃   ┃   ┗ sub-01_perceptionNaturalImageTest_VC_v2.h5
┃   ┃   ┃   ┗ sub-02_perceptionNaturalImageTest_VC_v2.h5
┃   ┃   ┃   ┗ sub-03_perceptionNaturalImageTest_VC_v2.h5
```

2. Download ImageNet (200 categories used in this work) by running `python download.py --save_path ./dataset/`. Please decompress them and put them in this repository as shown below.
```
┣ 📂 dataset
┃   ┣ 📂 ImageNet
┃   ┃   ┣ 📂 n01443537
┃   ┃   ┃   ┗ n01443537_2.JPEG
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📂 n01518878
┃   ┃   ┃   ┗ n01518878_2.JPEG
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📂 ...  
```

3. Download visual stimuli of [DIR dataset](https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577) (Kamitani Lab). Please download them and put them in this repository as shown below.
```
┣ 📂 dataset
┃   ┣ 📂 DIR
┃   ┃   ┣ 📂 stimulus
┃   ┃   ┃   📂 train
┃   ┃   ┃   ┃   ┗ *.JPEG
┃   ┃   ┃   ┃   ┗ ...
┃   ┃   ┃   📂 test
┃   ┃   ┃   ┃   ┗ *.JPEG 
┃   ┃   ┃   ┃   ┗ ...
```

4. (Optional) Download [THINGS-odd-one-out](https://osf.io/f5rn6/) and [THINGS object concept and object image database](https://osf.io/jum2f/)

5. (Optional) Download [DINOv2-small](https://huggingface.co/facebook/dinov2-small), [DINOv2-base](https://huggingface.co/facebook/dinov2-base), [DINOv2-large](https://huggingface.co/facebook/dinov2-large), [SimCLR-small](https://dl.fbaipublicfiles.com/slip/simclr_small_25ep.pt), [SimCLR-base](https://dl.fbaipublicfiles.com/slip/simclr_base_25ep.pt), and [SimCLR-large](https://dl.fbaipublicfiles.com/slip/simclr_large_25ep.pt).

## Feature extraction
CLIP
```
python extract_clip_features.py --model_size ViT-B/16 --extract_THINGS False
```
--model_size can be either ViT-B/16 or ViT-L/14.

SimCLR
```
python extract_simclr_features.py --model_size simclr_base_25ep --extract_THINGS False
```
--model_size can be simclr_small_25ep, simclr_base_25ep, or simclr_large_25ep.

DINOv2
```
python extract_dinov2_features.py --model_size dinov2-base --extract_THINGS False
```
--model_size can be dinov2-small, dinov2-base, or dinov2-large.

## Training Brain-in-the-loop supervised models
```
python brain_in_the_loop_training.py --proxy CLIP --model_size base --sub sub-3
```
## Repo organization
* `dataset`: Folder to dataset files
* `pretrained_features`: Folder to pretrained feature files
* `pretrained_models`: Folder to pretrained model files
* `data, modules`: Brain-in-the-loop supervision codes
* `smallcap`: SMALLCAP codes
* `download.py`: Codes for ImageNet download
* `extract_*_features.py`: Codes for extracting features
* `brain_in_the_loop_training.py`: Codes for training brain-in-the-loop supervised models
* `gpt_decoder_training.py`: Codes for training gpt-based semantic reconstruction models

## Acknowledgement
We thank Kamitani Lab for making their raw and pre-processed data public. Our semantic reconstruction model implementation is based on the [SMALLCAP](https://github.com/RitaRamo/smallcap). We thank these authors for making their codes available!